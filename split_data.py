from __future__ import annotations

import random
import shutil
from datetime import datetime
from pathlib import Path


# ==========================================
# 설정
# ==========================================

DATASET_DIR = Path("dataset")

TRAIN_IMAGE_DIR = DATASET_DIR / "train" / "images"
TRAIN_MASK_DIR = DATASET_DIR / "train" / "masks"

VAL_IMAGE_DIR = DATASET_DIR / "val" / "images"
VAL_MASK_DIR = DATASET_DIR / "val" / "masks"

TEST_IMAGE_DIR = DATASET_DIR / "test" / "images"
TEST_MASK_DIR = DATASET_DIR / "test" / "masks"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

RANDOM_SEED = 42


def find_images(folder: Path) -> list[Path]:
    """폴더 안의 이미지 파일을 찾아 정렬합니다."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_empty_split(folder: Path, split_name: str) -> None:
    """val/test 폴더가 이미 사용 중인지 검사합니다."""
    files = list(folder.iterdir())

    if files:
        raise RuntimeError(
            f"{split_name} 폴더가 비어 있지 않습니다: {folder}\n"
            "기존 파일을 확인한 뒤 비우고 다시 실행하세요."
        )


def move_pair(
    image_path: Path,
    destination_image_dir: Path,
    destination_mask_dir: Path,
) -> None:
    """이미지와 대응 마스크를 함께 이동합니다."""
    mask_path = TRAIN_MASK_DIR / f"{image_path.stem}.png"

    if not mask_path.exists():
        raise FileNotFoundError(
            f"다음 이미지의 마스크를 찾을 수 없습니다:\n"
            f"이미지: {image_path}\n"
            f"예상 마스크: {mask_path}"
        )

    destination_image_dir.mkdir(parents=True, exist_ok=True)
    destination_mask_dir.mkdir(parents=True, exist_ok=True)

    shutil.move(
        str(image_path),
        str(destination_image_dir / image_path.name),
    )

    shutil.move(
        str(mask_path),
        str(destination_mask_dir / mask_path.name),
    )


def main() -> None:
    # 필요한 폴더 검사
    if not TRAIN_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"train 이미지 폴더가 없습니다: {TRAIN_IMAGE_DIR}"
        )

    if not TRAIN_MASK_DIR.exists():
        raise FileNotFoundError(
            f"train 마스크 폴더가 없습니다: {TRAIN_MASK_DIR}"
        )

    VAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    VAL_MASK_DIR.mkdir(parents=True, exist_ok=True)
    TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    TEST_MASK_DIR.mkdir(parents=True, exist_ok=True)

    # 이미 분할된 데이터가 있으면 중단
    check_empty_split(VAL_IMAGE_DIR, "val/images")
    check_empty_split(VAL_MASK_DIR, "val/masks")
    check_empty_split(TEST_IMAGE_DIR, "test/images")
    check_empty_split(TEST_MASK_DIR, "test/masks")

    image_paths = find_images(TRAIN_IMAGE_DIR)

    number_of_images = len(image_paths)

    if number_of_images < 3:
        raise RuntimeError(
            "train/val/test로 나누려면 이미지가 최소 3장 필요합니다."
        )

    # 모든 이미지에 대응 마스크가 있는지 먼저 확인
    for image_path in image_paths:
        mask_path = TRAIN_MASK_DIR / f"{image_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"마스크가 없습니다: {mask_path}"
            )

    # 원본 데이터 백업
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"dataset_backup_before_split_{timestamp}")

    shutil.copytree(DATASET_DIR, backup_dir)

    print("=" * 60)
    print("데이터 백업 완료")
    print(f"백업 위치: {backup_dir.resolve()}")
    print("=" * 60)

    # 파일 순서를 무작위로 섞음
    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(image_paths)

    # 적은 데이터에서도 val/test가 최소 1장씩 있도록 설정
    number_of_val = max(1, round(number_of_images * 0.15))
    number_of_test = max(1, round(number_of_images * 0.15))
    number_of_train = (
        number_of_images
        - number_of_val
        - number_of_test
    )

    if number_of_train < 1:
        raise RuntimeError(
            "분할 후 train 데이터가 남지 않습니다."
        )

    val_images = image_paths[:number_of_val]

    test_images = image_paths[
        number_of_val:
        number_of_val + number_of_test
    ]

    remaining_train_images = image_paths[
        number_of_val + number_of_test:
    ]

    print("분할 예정")
    print("-" * 60)

    for image_path in remaining_train_images:
        print(f"TRAIN : {image_path.name}")

    for image_path in val_images:
        print(f"VAL   : {image_path.name}")

    for image_path in test_images:
        print(f"TEST  : {image_path.name}")

    print("-" * 60)
    print(f"Train: {len(remaining_train_images)}장")
    print(f"Val:   {len(val_images)}장")
    print(f"Test:  {len(test_images)}장")

    answer = input(
        "\n이대로 파일을 이동할까요? "
        "계속하려면 y를 입력하세요: "
    ).strip().lower()

    if answer != "y":
        print("분할을 취소했습니다.")
        return

    # val로 이동
    for image_path in val_images:
        move_pair(
            image_path,
            VAL_IMAGE_DIR,
            VAL_MASK_DIR,
        )

    # test로 이동
    for image_path in test_images:
        move_pair(
            image_path,
            TEST_IMAGE_DIR,
            TEST_MASK_DIR,
        )

    print("\n데이터 분할이 완료되었습니다.")
    print(f"Train: {len(remaining_train_images)}장")
    print(f"Val:   {len(val_images)}장")
    print(f"Test:  {len(test_images)}장")
    print()
    print("다음 명령어를 실행하세요:")
    print("python check_data.py")


if __name__ == "__main__":
    main()