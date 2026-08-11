from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone inference runner for Marshmello best_unet.pt"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to best_unet.pt",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image or directory containing images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("standalone_predictions"),
        help="Output directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override checkpoint threshold",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device",
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        default=None,
        help="Optional JSON result path. Default: <output>/result.json",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]

    if path.is_dir():
        return sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    raise FileNotFoundError(f"Input path does not exist: {path}")


def create_model(encoder_name: str) -> torch.nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(
    model_path: Path,
    device: torch.device,
    threshold_override: float | None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    # This script assumes the .pt file is trusted.
    # The project's checkpoint is a dictionary containing model_state_dict + metadata.
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected a checkpoint dictionary. "
            f"Got: {type(checkpoint).__name__}"
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        encoder_name = str(checkpoint.get("encoder_name", "resnet34"))
        image_size = int(checkpoint.get("image_size", 512))
        checkpoint_threshold = float(checkpoint.get("threshold", 0.5))
    else:
        # Fallback for a plain state_dict.
        state_dict = checkpoint
        encoder_name = "resnet34"
        image_size = 512
        checkpoint_threshold = 0.5

    if not isinstance(state_dict, dict):
        raise TypeError("model_state_dict is not a dictionary.")

    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else checkpoint_threshold
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1. Got: {threshold}")

    model = create_model(encoder_name)
    state_dict = strip_module_prefix(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load model weights. "
            "Check segmentation_models_pytorch version and encoder architecture.\n"
            f"Original error:\n{exc}"
        ) from exc

    model = model.to(device)
    model.eval()

    metadata = {
        "encoder_name": encoder_name,
        "image_size": image_size,
        "threshold": threshold,
        "epoch": checkpoint.get("epoch"),
        "val_dice": checkpoint.get("val_dice"),
    }
    return model, metadata


def letterbox_rgb(
    image_rgb: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    original_height, original_width = image_rgb.shape[:2]

    scale = min(
        image_size / original_width,
        image_size / original_height,
    )
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

    canvas[top:top + new_height, left:left + new_width] = resized

    meta = {
        "original_height": original_height,
        "original_width": original_width,
        "new_height": new_height,
        "new_width": new_width,
        "top": top,
        "left": left,
    }
    return canvas, meta


def rgb_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).unsqueeze(0).float()


def restore_probability(
    probability: np.ndarray,
    meta: dict[str, int],
) -> np.ndarray:
    top = meta["top"]
    left = meta["left"]
    new_height = meta["new_height"]
    new_width = meta["new_width"]

    cropped = probability[
        top:top + new_height,
        left:left + new_width,
    ]

    return cv2.resize(
        cropped,
        (meta["original_width"], meta["original_height"]),
        interpolation=cv2.INTER_LINEAR,
    )


def make_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image_rgb.copy()
    selected = mask.astype(bool)

    red = np.zeros_like(image_rgb)
    red[..., 0] = 255

    overlay[selected] = (
        (1.0 - alpha) * image_rgb[selected]
        + alpha * red[selected]
    ).astype(np.uint8)

    return overlay


def mask_bbox(mask: np.ndarray) -> dict[str, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    return {
        "x": x1,
        "y": y1,
        "width": x2 - x1 + 1,
        "height": y2 - y1 + 1,
    }


@torch.inference_mode()
def predict_one(
    model: torch.nn.Module,
    image_path: Path,
    output_dir: Path,
    device: torch.device,
    image_size: int,
    threshold: float,
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    letterboxed, letterbox_meta = letterbox_rgb(image_rgb, image_size)
    tensor = rgb_to_tensor(letterboxed).to(device)

    logits = model(tensor)
    probability_square = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    probability = restore_probability(probability_square, letterbox_meta)

    mask = probability >= threshold
    overlay_rgb = make_overlay(image_rgb, mask)

    mask_dir = output_dir / "masks"
    probability_dir = output_dir / "probabilities"
    overlay_dir = output_dir / "overlays"

    for directory in (mask_dir, probability_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mask_path = mask_dir / f"{image_path.stem}.png"
    probability_path = probability_dir / f"{image_path.stem}.png"
    overlay_path = overlay_dir / f"{image_path.stem}_overlay.jpg"

    mask_u8 = mask.astype(np.uint8) * 255
    probability_u8 = np.clip(probability * 255.0, 0, 255).astype(np.uint8)

    cv2.imwrite(str(mask_path), mask_u8)
    cv2.imwrite(str(probability_path), probability_u8)
    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
    )

    positive_pixels = int(mask.sum())
    total_pixels = int(mask.size)

    return {
        "image": str(image_path),
        "detected": bool(positive_pixels > 0),
        "positive_pixels": positive_pixels,
        "total_pixels": total_pixels,
        "coverage_ratio": float(positive_pixels / total_pixels),
        "coverage_percent": float(positive_pixels / total_pixels * 100.0),
        "mean_probability": float(probability.mean()),
        "max_probability": float(probability.max()),
        "bbox": mask_bbox(mask),
        "outputs": {
            "mask": str(mask_path),
            "probability": str(probability_path),
            "overlay": str(overlay_path),
        },
    }


def to_json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()

    device = choose_device(args.device)
    images = collect_images(args.input)
    if not images:
        raise RuntimeError(f"No supported images found in: {args.input}")

    model, model_meta = load_checkpoint(
        args.model,
        device,
        args.threshold,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    results = []
    for image_path in images:
        try:
            result = predict_one(
                model=model,
                image_path=image_path,
                output_dir=args.output,
                device=device,
                image_size=int(model_meta["image_size"]),
                threshold=float(model_meta["threshold"]),
            )
            results.append(result)
        except Exception as exc:
            results.append({
                "image": str(image_path),
                "error": f"{type(exc).__name__}: {exc}",
            })

    payload = {
        "model": str(args.model),
        "device": str(device),
        "checkpoint": {
            key: to_json_safe(value)
            for key, value in model_meta.items()
        },
        "results": results,
    }

    json_text = json.dumps(payload, ensure_ascii=False, indent=2)

    json_path = args.json_file or (args.output / "result.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")

    print(json_text)
    print(f"\nJSON saved to: {json_path}")


if __name__ == "__main__":
    main()
