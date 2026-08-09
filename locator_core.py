from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from locator_geometry import (
    compute_mask_shape_metrics,
    locate_navel_from_heatmap,
    navel_boundary_margin_ratio,
    significant_component_count,
)
from locator_settings import (
    ABDOMEN_THRESHOLD,
    ENCODER_NAME,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    NAVEL_CANDIDATE_PEAK_RATIO,
    NAVEL_HEATMAP_POS_WEIGHT,
    NAVEL_LOSS_WEIGHT,
    NAVEL_MIN_CANDIDATE_PROBABILITY,
    SEED,
    USE_IMAGENET_WEIGHTS,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class AnatomyPrediction:
    abdomen_mask: np.ndarray
    abdomen_probability: np.ndarray
    navel_probability: np.ndarray
    abdomen_polygon: list[tuple[int, int]]
    navel_x: int
    navel_y: int
    navel_confidence: float
    abdomen_confidence: float
    valid: bool
    warning: str
    abdomen_solidity: float
    abdomen_extent: float
    abdomen_bbox_aspect_ratio: float
    abdomen_core_width_ratio: float
    abdomen_component_count: int
    navel_boundary_margin_ratio: float
    navel_heatmap_concentration: float
    navel_heatmap_spread_ratio: float
    navel_candidate_area_ratio: float
    navel_candidate_component_count: int


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def mask_path(mask_dir: Path, image_path: Path) -> Path:
    return mask_dir / f"{image_path.stem}.png"


def letterbox_array(
    array: np.ndarray,
    image_size: int,
    interpolation: int,
    fill_value: int | float = 0,
) -> tuple[np.ndarray, dict[str, int]]:
    height, width = array.shape[:2]
    scale = min(image_size / width, image_size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(array, (new_width, new_height), interpolation=interpolation)

    if array.ndim == 3:
        canvas = np.full(
            (image_size, image_size, array.shape[2]),
            fill_value,
            dtype=array.dtype,
        )
    else:
        canvas = np.full((image_size, image_size), fill_value, dtype=array.dtype)

    left = (image_size - new_width) // 2
    top = (image_size - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    metadata = {
        "original_height": height,
        "original_width": width,
        "new_height": new_height,
        "new_width": new_width,
        "top": top,
        "left": left,
    }
    return canvas, metadata


def restore_to_original(
    square: np.ndarray,
    metadata: dict[str, int],
    interpolation: int,
) -> np.ndarray:
    top = metadata["top"]
    left = metadata["left"]
    new_height = metadata["new_height"]
    new_width = metadata["new_width"]
    cropped = square[top : top + new_height, left : left + new_width]
    return cv2.resize(
        cropped,
        (metadata["original_width"], metadata["original_height"]),
        interpolation=interpolation,
    )


def normalize_image(image_rgb: np.ndarray) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def _warp_affine_triplet(
    image_rgb: np.ndarray,
    abdomen: np.ndarray,
    navel_heatmap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    angle = random.uniform(-14.0, 14.0)
    scale = random.uniform(0.88, 1.12)
    translate_x = random.uniform(-0.08, 0.08) * width
    translate_y = random.uniform(-0.08, 0.08) * height

    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 2] += translate_x
    matrix[1, 2] += translate_y

    image_warped = cv2.warpAffine(
        image_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    abdomen_warped = cv2.warpAffine(
        abdomen,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    navel_warped = cv2.warpAffine(
        navel_heatmap,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return image_warped, abdomen_warped, navel_warped


def _warp_perspective_triplet(
    image_rgb: np.ndarray,
    abdomen: np.ndarray,
    navel_heatmap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    jitter = 0.035 * min(height, width)
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    destination = source + np.array(
        [
            [random.uniform(-jitter, jitter), random.uniform(-jitter, jitter)],
            [random.uniform(-jitter, jitter), random.uniform(-jitter, jitter)],
            [random.uniform(-jitter, jitter), random.uniform(-jitter, jitter)],
            [random.uniform(-jitter, jitter), random.uniform(-jitter, jitter)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)

    image_warped = cv2.warpPerspective(
        image_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    abdomen_warped = cv2.warpPerspective(
        abdomen,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    navel_warped = cv2.warpPerspective(
        navel_heatmap,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return image_warped, abdomen_warped, navel_warped


def augment_anatomy_sample(
    image_rgb: np.ndarray,
    abdomen: np.ndarray,
    navel_heatmap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply geometry-consistent augmentation to image, abdomen and heatmap."""
    original_image = image_rgb
    original_abdomen = abdomen
    original_navel = navel_heatmap

    if random.random() < 0.5:
        image_rgb = np.ascontiguousarray(image_rgb[:, ::-1])
        abdomen = np.ascontiguousarray(abdomen[:, ::-1])
        navel_heatmap = np.ascontiguousarray(navel_heatmap[:, ::-1])

    if random.random() < 0.80:
        image_rgb, abdomen, navel_heatmap = _warp_affine_triplet(
            image_rgb, abdomen, navel_heatmap
        )

    if random.random() < 0.25:
        image_rgb, abdomen, navel_heatmap = _warp_perspective_triplet(
            image_rgb, abdomen, navel_heatmap
        )

    # 지나친 기하 변환으로 라벨이 거의 사라졌다면 해당 샘플은 원본 기하를 사용합니다.
    if np.count_nonzero(abdomen) < max(32, int(0.45 * np.count_nonzero(original_abdomen))):
        image_rgb = original_image.copy()
        abdomen = original_abdomen.copy()
        navel_heatmap = original_navel.copy()
    elif float(np.max(navel_heatmap)) < 64.0:
        image_rgb = original_image.copy()
        abdomen = original_abdomen.copy()
        navel_heatmap = original_navel.copy()

    if random.random() < 0.70:
        alpha = random.uniform(0.82, 1.18)
        beta = random.uniform(-20.0, 20.0)
        image_rgb = np.clip(
            image_rgb.astype(np.float32) * alpha + beta,
            0,
            255,
        ).astype(np.uint8)

    if random.random() < 0.25:
        gamma = random.uniform(0.80, 1.20)
        lookup = np.array(
            [((value / 255.0) ** gamma) * 255.0 for value in range(256)],
            dtype=np.uint8,
        )
        image_rgb = cv2.LUT(image_rgb, lookup)

    if random.random() < 0.20:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.82, 1.18)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        image_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if random.random() < 0.15:
        kernel = random.choice((3, 5))
        image_rgb = cv2.GaussianBlur(image_rgb, (kernel, kernel), 0)

    return (
        np.ascontiguousarray(image_rgb),
        np.ascontiguousarray(abdomen),
        np.ascontiguousarray(navel_heatmap),
    )


class AnatomyDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        abdomen_mask_dir: Path,
        navel_mask_dir: Path,
        train: bool,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.image_dir = image_dir
        self.abdomen_mask_dir = abdomen_mask_dir
        self.navel_mask_dir = navel_mask_dir
        self.train = train
        self.image_size = image_size
        self.images = list_images(image_dir)
        if not self.images:
            raise RuntimeError(f"이미지가 없습니다: {image_dir}")

        for image_path in self.images:
            abdomen_path = mask_path(abdomen_mask_dir, image_path)
            navel_path = mask_path(navel_mask_dir, image_path)
            if not abdomen_path.exists():
                raise FileNotFoundError(f"복부 마스크가 없습니다: {abdomen_path}")
            if not navel_path.exists():
                raise FileNotFoundError(f"배꼽 heatmap이 없습니다: {navel_path}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.images[index]
        abdomen_path = mask_path(self.abdomen_mask_dir, image_path)
        navel_path = mask_path(self.navel_mask_dir, image_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        abdomen = cv2.imread(str(abdomen_path), cv2.IMREAD_GRAYSCALE)
        navel_heatmap = cv2.imread(str(navel_path), cv2.IMREAD_GRAYSCALE)

        if image_bgr is None or abdomen is None or navel_heatmap is None:
            raise RuntimeError(f"파일을 읽지 못했습니다: {image_path.name}")
        if (
            image_bgr.shape[:2] != abdomen.shape[:2]
            or image_bgr.shape[:2] != navel_heatmap.shape[:2]
        ):
            raise RuntimeError(f"이미지와 마스크/heatmap 크기가 다릅니다: {image_path.name}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.train:
            image_rgb, abdomen, navel_heatmap = augment_anatomy_sample(
                image_rgb,
                abdomen,
                navel_heatmap,
            )

        image_rgb, _ = letterbox_array(
            image_rgb,
            self.image_size,
            interpolation=cv2.INTER_AREA,
        )
        abdomen, _ = letterbox_array(
            abdomen,
            self.image_size,
            interpolation=cv2.INTER_NEAREST,
        )
        navel_heatmap, _ = letterbox_array(
            navel_heatmap,
            self.image_size,
            interpolation=cv2.INTER_LINEAR,
        )

        image_tensor = normalize_image(image_rgb)
        abdomen_tensor = torch.from_numpy((abdomen > 127).astype(np.float32))
        navel_tensor = torch.from_numpy(
            np.clip(navel_heatmap.astype(np.float32) / 255.0, 0.0, 1.0)
        )
        target = torch.stack([abdomen_tensor, navel_tensor], dim=0)
        return image_tensor, target, image_path.name


def create_locator_model(
    encoder_name: str = ENCODER_NAME,
    use_imagenet_weights: bool = USE_IMAGENET_WEIGHTS,
) -> torch.nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if use_imagenet_weights else None,
        in_channels=3,
        classes=2,
        activation=None,
    )


class AnatomyLoss(torch.nn.Module):
    def __init__(
        self,
        navel_weight: float = NAVEL_LOSS_WEIGHT,
        navel_positive_weight: float = NAVEL_HEATMAP_POS_WEIGHT,
    ) -> None:
        super().__init__()
        self.navel_weight = float(navel_weight)
        self.navel_positive_weight = float(max(0.0, navel_positive_weight))
        self.abdomen_bce = torch.nn.BCEWithLogitsLoss()
        self.abdomen_dice = smp.losses.DiceLoss(
            mode="binary",
            from_logits=True,
            smooth=1.0,
        )

    def abdomen_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.abdomen_bce(logits, target) + self.abdomen_dice(logits, target)

    def navel_heatmap_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.clamp(0.0, 1.0)
        probability = torch.sigmoid(logits)
        positive_weight = 1.0 + self.navel_positive_weight * target

        # Gaussian heatmap의 넓은 0 영역이 loss를 압도하지 않도록 focal factor를 적용합니다.
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = target * probability + (1.0 - target) * (1.0 - probability)
        focal_factor = (1.0 - pt).pow(2.0)
        focal_bce = (bce * focal_factor * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)

        mse = ((probability - target).pow(2.0) * positive_weight).sum()
        mse = mse / positive_weight.sum().clamp_min(1.0)
        return 0.65 * focal_bce + 0.35 * mse

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        abdomen_loss = self.abdomen_loss(logits[:, 0:1], targets[:, 0:1])
        navel_loss = self.navel_heatmap_loss(logits[:, 1:2], targets[:, 1:2])
        return abdomen_loss + self.navel_weight * navel_loss


def _soft_coordinates(heatmap: torch.Tensor, power: float = 8.0) -> tuple[torch.Tensor, torch.Tensor]:
    batch, height, width = heatmap.shape
    weights = heatmap.clamp_min(0.0).pow(power)
    denominator = weights.sum(dim=(1, 2)).clamp_min(1e-8)
    xs = torch.arange(width, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, width)
    ys = torch.arange(height, device=heatmap.device, dtype=heatmap.dtype).view(1, height, 1)
    x = (weights * xs).sum(dim=(1, 2)) / denominator
    y = (weights * ys).sum(dim=(1, 2)) / denominator
    return x, y


@torch.no_grad()
def locator_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    abdomen_threshold: float = ABDOMEN_THRESHOLD,
) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    abdomen_pred = probabilities[:, 0:1] >= abdomen_threshold
    abdomen_true = targets[:, 0:1] >= 0.5
    dims = (1, 2, 3)
    tp = (abdomen_pred & abdomen_true).sum(dim=dims).float()
    fp = (abdomen_pred & ~abdomen_true).sum(dim=dims).float()
    fn = (~abdomen_pred & abdomen_true).sum(dim=dims).float()
    eps = 1e-7
    abdomen_dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    navel_probability = probabilities[:, 1]
    navel_true = targets[:, 1]
    pred_x, pred_y = _soft_coordinates(navel_probability)
    true_x, true_y = _soft_coordinates(navel_true)
    distances = torch.sqrt((pred_x - true_x).pow(2.0) + (pred_y - true_y).pow(2.0))
    height, width = navel_probability.shape[-2:]
    diagonal = float((height**2 + width**2) ** 0.5)
    normalized_distance = distances / max(diagonal, 1.0)

    return {
        "abdomen_dice": float(abdomen_dice.mean().item()),
        "navel_distance_px": float(distances.mean().item()),
        "navel_distance_normalized": float(normalized_distance.mean().item()),
        "navel_peak_confidence": float(navel_probability.amax(dim=(1, 2)).mean().item()),
    }


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    number, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if number <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return (labels == largest_label).astype(np.uint8) * 255


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def postprocess_abdomen_mask(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    height, width = binary.shape[:2]
    kernel_size = int(round(min(height, width) * 0.02))
    kernel_size = int(np.clip(kernel_size, 7, 15))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = keep_largest_component(binary)
    binary = fill_mask_holes(binary)
    return binary


def mask_to_polygon(mask: np.ndarray, epsilon_ratio: float = 0.008) -> list[tuple[int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
    return [(int(point[0][0]), int(point[0][1])) for point in polygon]


def predict_anatomy(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    abdomen_threshold: float,
) -> AnatomyPrediction:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    letterboxed, metadata = letterbox_array(
        image_rgb,
        image_size,
        interpolation=cv2.INTER_AREA,
    )
    tensor = normalize_image(letterboxed).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()

    abdomen_probability = restore_to_original(
        probabilities[0],
        metadata,
        interpolation=cv2.INTER_LINEAR,
    )
    navel_probability = restore_to_original(
        probabilities[1],
        metadata,
        interpolation=cv2.INTER_LINEAR,
    )
    abdomen_probability = np.clip(abdomen_probability, 0.0, 1.0).astype(np.float32)
    navel_probability = np.clip(navel_probability, 0.0, 1.0).astype(np.float32)

    raw_abdomen = (abdomen_probability >= abdomen_threshold).astype(np.uint8) * 255
    raw_component_count = significant_component_count(raw_abdomen)
    abdomen_mask = postprocess_abdomen_mask(raw_abdomen)
    shape = compute_mask_shape_metrics(abdomen_mask)

    valid = True
    warnings: list[str] = []
    if shape.area_ratio < 0.08:
        valid = False
        warnings.append("복부 영역이 너무 작게 검출됨")
    if shape.area_ratio > 0.92:
        valid = False
        warnings.append("복부 영역이 사진 대부분을 차지함")
    if shape.solidity < 0.40 and shape.area_ratio >= 0.08:
        valid = False
        warnings.append("복부 영역 형태가 지나치게 불규칙함")

    navel = locate_navel_from_heatmap(
        navel_probability,
        abdomen_mask,
        peak_ratio=NAVEL_CANDIDATE_PEAK_RATIO,
        minimum_probability=NAVEL_MIN_CANDIDATE_PROBABILITY,
    )
    navel_x = int(navel.x)
    navel_y = int(navel.y)
    navel_confidence = float(navel.peak_confidence)

    height, width = abdomen_mask.shape
    if not (0 <= navel_x < width and 0 <= navel_y < height):
        valid = False
        warnings.append("배꼽 좌표가 이미지 범위를 벗어남")
    elif abdomen_mask[navel_y, navel_x] == 0:
        valid = False
        warnings.append("배꼽 좌표가 복부 영역 밖에 있음")
    if navel_confidence <= 0.0:
        valid = False
        warnings.append("배꼽 heatmap에서 유효한 peak를 찾지 못함")

    selected_probabilities = abdomen_probability[abdomen_mask > 0]
    abdomen_confidence = (
        float(selected_probabilities.mean()) if selected_probabilities.size else 0.0
    )

    polygon = mask_to_polygon(abdomen_mask)
    if len(polygon) < 3:
        valid = False
        warnings.append("복부 외곽 다각형 생성 실패")

    boundary_margin = navel_boundary_margin_ratio(abdomen_mask, navel_x, navel_y)

    return AnatomyPrediction(
        abdomen_mask=abdomen_mask,
        abdomen_probability=abdomen_probability,
        navel_probability=navel_probability,
        abdomen_polygon=polygon,
        navel_x=navel_x,
        navel_y=navel_y,
        navel_confidence=navel_confidence,
        abdomen_confidence=abdomen_confidence,
        valid=valid,
        warning="; ".join(warnings),
        abdomen_solidity=float(shape.solidity),
        abdomen_extent=float(shape.extent),
        abdomen_bbox_aspect_ratio=float(shape.bbox_aspect_ratio),
        abdomen_core_width_ratio=float(shape.core_width_ratio),
        abdomen_component_count=int(raw_component_count),
        navel_boundary_margin_ratio=float(boundary_margin),
        navel_heatmap_concentration=float(navel.concentration),
        navel_heatmap_spread_ratio=float(navel.spread_ratio),
        navel_candidate_area_ratio=float(navel.candidate_area_ratio),
        navel_candidate_component_count=int(navel.candidate_component_count),
    )
