from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from settings import IMAGE_EXTENSIONS, IMAGE_SIZE, SEED

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def mask_path_for_image(mask_dir: Path, image_path: Path) -> Path:
    preferred = mask_dir / f"{image_path.stem}.png"
    if preferred.exists():
        return preferred

    for extension in IMAGE_EXTENSIONS:
        candidate = mask_dir / f"{image_path.stem}{extension}"
        if candidate.exists():
            return candidate

    return preferred


def collect_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in list_images(image_dir):
        mask_path = mask_path_for_image(mask_dir, image_path)
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def patient_id_from_stem(stem: str) -> str:
    # P049_01 -> P049
    # 파일명 규칙이 다르면 이 함수만 수정하세요.
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def build_transform(train: bool, image_size: int = IMAGE_SIZE) -> A.Compose:
    transforms: list[A.BasicTransform] = [
        A.LongestMaxSize(
            max_size=image_size,
            interpolation=cv2.INTER_AREA,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
        ),
    ]

    if train:
        # 현재는 임신선 분할만 학습하므로 좌우 반전이 가능합니다.
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.10,
                    contrast_limit=0.10,
                    p=0.30,
                ),
            ]
        )

    transforms.extend(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


class StretchMarkDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        train: bool,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = build_transform(train=train, image_size=image_size)
        self.pairs = collect_pairs(image_dir, mask_dir)

        if not self.pairs:
            raise RuntimeError(
                f"이미지-마스크 쌍을 찾지 못했습니다.\n"
                f"이미지 폴더: {image_dir}\n마스크 폴더: {mask_dir}"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path, mask_path = self.pairs[index]

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image_bgr is None:
            raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")
        if mask is None:
            raise RuntimeError(f"마스크를 읽지 못했습니다: {mask_path}")
        if image_bgr.shape[:2] != mask.shape[:2]:
            raise RuntimeError(
                f"이미지와 마스크 크기가 다릅니다: {image_path.name} "
                f"image={image_bgr.shape[:2]}, mask={mask.shape[:2]}"
            )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.uint8)

        result = self.transform(image=image_rgb, mask=mask)
        image_tensor = result["image"].float()
        mask_tensor = result["mask"].float().unsqueeze(0)

        return image_tensor, mask_tensor, image_path.name


def create_model(
    encoder_name: str,
    use_imagenet_weights: bool,
) -> torch.nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if use_imagenet_weights else None,
        in_channels=3,
        classes=1,
        activation=None,
    )


class CombinedLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(
            mode="binary",
            from_logits=True,
            smooth=1.0,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice(logits, targets)


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    targets_bool = targets >= 0.5

    dims = (1, 2, 3)
    tp = (predictions & targets_bool).sum(dim=dims).float()
    fp = (predictions & ~targets_bool).sum(dim=dims).float()
    fn = (~predictions & targets_bool).sum(dim=dims).float()

    eps = 1e-7
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    image = image * std + mean
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def make_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image_rgb.copy()
    selected = mask > 0
    red = np.zeros_like(image_rgb)
    red[..., 0] = 255
    overlay[selected] = (
        (1.0 - alpha) * image_rgb[selected] + alpha * red[selected]
    ).astype(np.uint8)
    return overlay


def letterbox_rgb(
    image_rgb: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    original_height, original_width = image_rgb.shape[:2]
    scale = min(image_size / original_width, image_size / original_height)
    new_width = max(1, round(original_width * scale))
    new_height = max(1, round(original_height * scale))

    resized = cv2.resize(
        image_rgb,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    left = (image_size - new_width) // 2
    top = (image_size - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized

    metadata = {
        "original_height": original_height,
        "original_width": original_width,
        "new_height": new_height,
        "new_width": new_width,
        "top": top,
        "left": left,
    }
    return canvas, metadata


def rgb_to_model_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    image = (image - mean) / std
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def restore_probability_to_original(
    probability: np.ndarray,
    metadata: dict[str, int],
) -> np.ndarray:
    top = metadata["top"]
    left = metadata["left"]
    new_height = metadata["new_height"]
    new_width = metadata["new_width"]

    cropped = probability[top : top + new_height, left : left + new_width]
    restored = cv2.resize(
        cropped,
        (metadata["original_width"], metadata["original_height"]),
        interpolation=cv2.INTER_LINEAR,
    )
    return restored


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
