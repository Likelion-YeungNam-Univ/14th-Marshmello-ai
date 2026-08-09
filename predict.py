from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from settings import CHECKPOINT_DIR, IMAGE_EXTENSIONS, NEW_IMAGES_DIR, OUTPUT_DIR
from unet_core import (
    create_model,
    letterbox_rgb,
    make_overlay,
    restore_probability_to_original,
    rgb_to_model_tensor,
)


def collect_input_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            file
            for file in path.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="새 복부 사진에서 임신선 마스크 예측")
    parser.add_argument("--input", type=Path, default=NEW_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "predictions")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINT_DIR / "best_unet.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("먼저 python train.py를 실행해 best_unet.pt를 만드세요.")

    input_images = collect_input_images(args.input)
    if not input_images:
        args.input.mkdir(parents=True, exist_ok=True)
        raise RuntimeError(
            f"예측할 이미지가 없습니다. 다음 폴더에 jpg/png를 넣으세요: {args.input}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 512))
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("threshold", 0.5))
    )
    encoder_name = str(checkpoint.get("encoder_name", "resnet34"))

    model = create_model(encoder_name, use_imagenet_weights=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mask_dir = args.output / "masks"
    probability_dir = args.output / "probabilities"
    overlay_dir = args.output / "overlays"
    for directory in (mask_dir, probability_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for image_path in input_images:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"읽기 실패: {image_path}")
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            letterboxed, metadata = letterbox_rgb(image_rgb, image_size)
            tensor = rgb_to_model_tensor(letterboxed).to(device)

            logits = model(tensor)
            probability_512 = torch.sigmoid(logits)[0, 0].cpu().numpy()
            probability = restore_probability_to_original(probability_512, metadata)
            mask = probability >= threshold

            probability_u8 = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
            mask_u8 = mask.astype(np.uint8) * 255
            overlay_rgb = make_overlay(image_rgb, mask)

            cv2.imwrite(str(mask_dir / f"{image_path.stem}.png"), mask_u8)
            cv2.imwrite(
                str(probability_dir / f"{image_path.stem}.png"), probability_u8
            )
            cv2.imwrite(
                str(overlay_dir / f"{image_path.stem}_overlay.jpg"),
                cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
            )
            print(f"완료: {image_path.name}")

    print(f"\n예측 완료: {args.output}")
    print(f"사용 threshold: {threshold}")


if __name__ == "__main__":
    main()
