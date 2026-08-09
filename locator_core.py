from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import Dataset

from locator_settings import (
    ABDOMEN_THRESHOLD,
    ENCODER_NAME,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    NAVEL_LOSS_WEIGHT,
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
    fill_value: int = 0,
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
                raise FileNotFoundError(f"배꼽 마스크가 없습니다: {navel_path}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.images[index]
        abdomen_path = mask_path(self.abdomen_mask_dir, image_path)
        navel_path = mask_path(self.navel_mask_dir, image_path)

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        abdomen = cv2.imread(str(abdomen_path), cv2.IMREAD_GRAYSCALE)
        navel = cv2.imread(str(navel_path), cv2.IMREAD_GRAYSCALE)

        if image_bgr is None or abdomen is None or navel is None:
            raise RuntimeError(f"파일을 읽지 못했습니다: {image_path.name}")
        if image_bgr.shape[:2] != abdomen.shape[:2] or image_bgr.shape[:2] != navel.shape[:2]:
            raise RuntimeError(f"이미지와 마스크 크기가 다릅니다: {image_path.name}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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
        navel, _ = letterbox_array(
            navel,
            self.image_size,
            interpolation=cv2.INTER_NEAREST,
        )

        if self.train and random.random() < 0.5:
            image_rgb = np.ascontiguousarray(image_rgb[:, ::-1])
            abdomen = np.ascontiguousarray(abdomen[:, ::-1])
            navel = np.ascontiguousarray(navel[:, ::-1])

        if self.train and random.random() < 0.35:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-12.0, 12.0)
            image_rgb = np.clip(
                image_rgb.astype(np.float32) * alpha + beta,
                0,
                255,
            ).astype(np.uint8)

        image_tensor = normalize_image(image_rgb)
        abdomen_tensor = torch.from_numpy((abdomen > 127).astype(np.float32))
        navel_tensor = torch.from_numpy((navel > 127).astype(np.float32))
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
    def __init__(self, navel_weight: float = NAVEL_LOSS_WEIGHT) -> None:
        super().__init__()
        self.navel_weight = navel_weight
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(
            mode="binary",
            from_logits=True,
            smooth=1.0,
        )

    def channel_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, target) + self.dice(logits, target)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        abdomen_loss = self.channel_loss(logits[:, 0:1], targets[:, 0:1])
        navel_loss = self.channel_loss(logits[:, 1:2], targets[:, 1:2])
        return abdomen_loss + self.navel_weight * navel_loss


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

    navel_prob = probabilities[:, 1]
    navel_true = targets[:, 1]
    batch_size, height, width = navel_prob.shape
    distances: list[torch.Tensor] = []

    for index in range(batch_size):
        pred_flat = torch.argmax(navel_prob[index])
        true_flat = torch.argmax(navel_true[index])
        pred_y = pred_flat // width
        pred_x = pred_flat % width
        true_y = true_flat // width
        true_x = true_flat % width
        distance = torch.sqrt(
            (pred_x.float() - true_x.float()) ** 2
            + (pred_y.float() - true_y.float()) ** 2
        )
        distances.append(distance)

    mean_distance = torch.stack(distances).mean() if distances else torch.tensor(0.0)
    normalized_distance = mean_distance / float((height**2 + width**2) ** 0.5)

    return {
        "abdomen_dice": float(abdomen_dice.mean().item()),
        "navel_distance_px": float(mean_distance.item()),
        "navel_distance_normalized": float(normalized_distance.item()),
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
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

    raw_abdomen = (abdomen_probability >= abdomen_threshold).astype(np.uint8) * 255
    abdomen_mask = postprocess_abdomen_mask(raw_abdomen)
    height, width = abdomen_mask.shape
    area_ratio = float(np.count_nonzero(abdomen_mask)) / float(height * width)

    valid = True
    warnings: list[str] = []
    if area_ratio < 0.08:
        valid = False
        warnings.append("복부 영역이 너무 작게 검출됨")
    if area_ratio > 0.92:
        valid = False
        warnings.append("복부 영역이 사진 대부분을 차지함")

    constrained = navel_probability.copy()
    if np.any(abdomen_mask > 0):
        constrained[abdomen_mask == 0] = 0.0

    flat_index = int(np.argmax(constrained))
    navel_y, navel_x = np.unravel_index(flat_index, constrained.shape)
    navel_confidence = float(constrained[navel_y, navel_x])

    if abdomen_mask[navel_y, navel_x] == 0:
        valid = False
        warnings.append("배꼽 좌표가 복부 영역 밖에 있음")

    selected_probabilities = abdomen_probability[abdomen_mask > 0]
    abdomen_confidence = (
        float(selected_probabilities.mean()) if selected_probabilities.size else 0.0
    )

    polygon = mask_to_polygon(abdomen_mask)
    if len(polygon) < 3:
        valid = False
        warnings.append("복부 외곽 다각형 생성 실패")

    return AnatomyPrediction(
        abdomen_mask=abdomen_mask,
        abdomen_probability=abdomen_probability,
        navel_probability=navel_probability,
        abdomen_polygon=polygon,
        navel_x=int(navel_x),
        navel_y=int(navel_y),
        navel_confidence=navel_confidence,
        abdomen_confidence=abdomen_confidence,
        valid=valid,
        warning="; ".join(warnings),
    )
