from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from settings import (
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
)
from unet_core import list_images, mask_path_for_image, patient_id_from_stem

SPLITS = {
    "train": (TRAIN_IMAGE_DIR, TRAIN_MASK_DIR),
    "val": (VAL_IMAGE_DIR, VAL_MASK_DIR),
    "test": (TEST_IMAGE_DIR, TEST_MASK_DIR),
}


def inspect_split(name: str, image_dir: Path, mask_dir: Path) -> tuple[int, set[str], int]:
    print(f"\n[{name.upper()}]")
    errors = 0
    patient_ids: set[str] = set()

    if not image_dir.exists():
        print(f"  오류: 이미지 폴더가 없습니다: {image_dir}")
        return 0, patient_ids, 1
    if not mask_dir.exists():
        print(f"  오류: 마스크 폴더가 없습니다: {mask_dir}")
        return 0, patient_ids, 1

    image_paths = list_images(image_dir)
    mask_files = list_images(mask_dir)
    image_stems = {path.stem for path in image_paths}
    mask_stems = {path.stem for path in mask_files}

    print(f"  이미지: {len(image_paths)}장")
    print(f"  마스크: {len(mask_files)}장")

    missing_masks = sorted(image_stems - mask_stems)
    extra_masks = sorted(mask_stems - image_stems)

    if missing_masks:
        errors += len(missing_masks)
        print("  오류: 대응 마스크가 없는 이미지:", ", ".join(missing_masks))
    if extra_masks:
        errors += len(extra_masks)
        print("  오류: 대응 이미지가 없는 마스크:", ", ".join(extra_masks))

    empty_masks = 0
    non_binary_masks = 0

    for image_path in image_paths:
        patient_ids.add(patient_id_from_stem(image_path.stem))
        mask_path = mask_path_for_image(mask_dir, image_path)
        if not mask_path.exists():
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"  오류: 이미지 읽기 실패: {image_path.name}")
            errors += 1
            continue
        if mask is None:
            print(f"  오류: 마스크 읽기 실패: {mask_path.name}")
            errors += 1
            continue
        if image.shape[:2] != mask.shape[:2]:
            print(
                f"  오류: 크기 불일치 {image_path.name} "
                f"image={image.shape[:2]} mask={mask.shape[:2]}"
            )
            errors += 1

        foreground_ratio = float((mask > 127).mean())
        if foreground_ratio == 0.0:
            empty_masks += 1

        unique_values = np.unique(mask)
        if not set(unique_values.tolist()).issubset({0, 255}):
            non_binary_masks += 1

    if empty_masks:
        print(f"  참고: 완전히 빈 마스크 {empty_masks}장 (임신선이 없는 사진이면 정상)")
    if non_binary_masks:
        print(
            f"  주의: 0/255 외 픽셀값이 있는 마스크 {non_binary_masks}장 "
            "(학습 시 127 기준으로 이진화됩니다)"
        )

    if not image_paths:
        print("  비어 있음")

    return len(image_paths), patient_ids, errors


def main() -> None:
    total_errors = 0
    patient_sets: dict[str, set[str]] = {}

    print("데이터 구조 검사 시작")
    for split_name, (image_dir, mask_dir) in SPLITS.items():
        _, patient_ids, errors = inspect_split(split_name, image_dir, mask_dir)
        patient_sets[split_name] = patient_ids
        total_errors += errors

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(patient_sets[left] & patient_sets[right])
        if overlap:
            total_errors += len(overlap)
            print(
                f"\n오류: {left}과 {right}에 같은 환자가 있습니다: "
                + ", ".join(overlap)
            )

    print("\n" + "=" * 60)
    if total_errors:
        print(f"검사 실패: 오류 {total_errors}개를 먼저 고쳐주세요.")
        sys.exit(1)

    print("검사 통과: 발견된 구조 오류가 없습니다.")
    if not list_images(VAL_IMAGE_DIR) or not list_images(TEST_IMAGE_DIR):
        print("다음 단계: val/test가 비어 있으므로 python split_data.py 를 실행하세요.")


if __name__ == "__main__":
    main()
