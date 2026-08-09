from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from settings import DATASET_DIR, OUTPUT_DIR
from unet_core import collect_pairs, make_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="정답 마스크가 원본과 맞는지 겹쳐 보기")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    args = parser.parse_args()

    image_dir = DATASET_DIR / args.split / "images"
    mask_dir = DATASET_DIR / args.split / "masks"
    output_dir = OUTPUT_DIR / "data_overlays" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(image_dir, mask_dir)
    if not pairs:
        raise RuntimeError(f"{args.split}에 이미지-마스크 쌍이 없습니다.")

    for image_path, mask_path in pairs:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            print(f"건너뜀: {image_path.name}")
            continue
        if image_bgr.shape[:2] != mask.shape[:2]:
            print(f"크기 불일치로 건너뜀: {image_path.name}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = make_overlay(image_rgb, mask > 127)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_overlay.jpg"), overlay_bgr)

    print(f"완료: {output_dir}")


if __name__ == "__main__":
    main()
