from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from locator_settings import (
    IMAGE_EXTENSIONS,
    LOCATOR_DATASET_DIR,
    MAX_NAVEL_RADIUS,
    MIN_NAVEL_RADIUS,
    NAVEL_RADIUS_RATIO,
    SEED,
)


def load_rois(path: Path) -> dict[str, list[tuple[int, int]]]:
    if not path.exists():
        raise FileNotFoundError(f"복부 ROI 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    result: dict[str, list[tuple[int, int]]] = {}
    for name, points in raw.items():
        parsed: list[tuple[int, int]] = []
        if not isinstance(points, list):
            continue
        for point in points:
            if isinstance(point, list) and len(point) == 2:
                parsed.append((int(point[0]), int(point[1])))
        if len(parsed) >= 3:
            result[str(name)] = parsed
    return result


def load_navel_points(path: Path) -> dict[str, tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"배꼽 좌표 파일이 없습니다: {path}")

    result: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                result[row["image_name"]] = (int(row["navel_x"]), int(row["navel_y"]))
            except (KeyError, TypeError, ValueError):
                continue
    return result


def collect_images(directories: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for directory in directories:
        if not directory.exists():
            raise FileNotFoundError(f"이미지 폴더가 없습니다: {directory}")
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.name in result:
                raise RuntimeError(
                    f"서로 다른 폴더에 같은 파일명이 있습니다: {path.name}\n"
                    "파일명을 고유하게 바꿔주세요."
                )
            result[path.name] = path
    return result


def create_masks(
    image_path: Path,
    roi_points: list[tuple[int, int]],
    navel_point: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")

    height, width = image.shape[:2]
    abdomen = np.zeros((height, width), dtype=np.uint8)
    polygon = np.asarray(roi_points, dtype=np.int32)
    cv2.fillPoly(abdomen, [polygon], 255)

    navel = np.zeros((height, width), dtype=np.uint8)
    x, y = navel_point
    x = min(max(int(x), 0), width - 1)
    y = min(max(int(y), 0), height - 1)
    radius = round(min(height, width) * NAVEL_RADIUS_RATIO)
    radius = max(MIN_NAVEL_RADIUS, min(radius, MAX_NAVEL_RADIUS))
    cv2.circle(navel, (x, y), radius, 255, thickness=cv2.FILLED)

    return abdomen, navel


def save_sample(
    split: str,
    image_path: Path,
    abdomen: np.ndarray,
    navel: np.ndarray,
) -> None:
    image_dir = LOCATOR_DATASET_DIR / split / "images"
    abdomen_dir = LOCATOR_DATASET_DIR / split / "abdomen_masks"
    navel_dir = LOCATOR_DATASET_DIR / split / "navel_masks"
    for directory in (image_dir, abdomen_dir, navel_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, image_dir / image_path.name)
    if not cv2.imwrite(str(abdomen_dir / f"{image_path.stem}.png"), abdomen):
        raise RuntimeError(f"복부 마스크 저장 실패: {image_path.name}")
    if not cv2.imwrite(str(navel_dir / f"{image_path.stem}.png"), navel):
        raise RuntimeError(f"배꼽 마스크 저장 실패: {image_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="수동 복부 ROI와 배꼽 좌표를 자동 위치 모델 학습 데이터로 변환합니다."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        nargs="+",
        default=[Path("new_images")],
        help="라벨링한 원본 이미지 폴더. 여러 폴더를 공백으로 구분해 지정할 수 있습니다.",
    )
    parser.add_argument("--abdomen-roi-json", type=Path, default=Path("abdomen_rois.json"))
    parser.add_argument("--navel-csv", type=Path, default=Path("navel_points.csv"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.05 <= args.val_ratio <= 0.5:
        raise ValueError("--val-ratio는 0.05~0.5 사이여야 합니다.")

    if LOCATOR_DATASET_DIR.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"이미 폴더가 있습니다: {LOCATOR_DATASET_DIR}\n"
                "다시 만들려면 --overwrite 옵션을 사용하세요."
            )
        shutil.rmtree(LOCATOR_DATASET_DIR)

    images = collect_images(args.image_dir)
    rois = load_rois(args.abdomen_roi_json)
    navels = load_navel_points(args.navel_csv)

    eligible_names = sorted(set(images) & set(rois) & set(navels))
    missing_roi = sorted(set(images) - set(rois))
    missing_navel = sorted(set(images) - set(navels))

    if len(eligible_names) < 3:
        raise RuntimeError(
            "복부 ROI와 배꼽 좌표가 모두 있는 이미지가 최소 3장 필요합니다.\n"
            f"현재 사용 가능: {len(eligible_names)}장"
        )

    rng = random.Random(SEED)
    rng.shuffle(eligible_names)
    val_count = max(1, round(len(eligible_names) * args.val_ratio))
    val_names = set(eligible_names[:val_count])

    manifest_rows: list[dict[str, str]] = []
    for name in eligible_names:
        split = "val" if name in val_names else "train"
        abdomen, navel = create_masks(images[name], rois[name], navels[name])
        save_sample(split, images[name], abdomen, navel)
        manifest_rows.append({"image_name": name, "split": split})

    manifest_path = LOCATOR_DATASET_DIR / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_name", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    train_count = len(eligible_names) - val_count
    print("=" * 60)
    print("자동 위치 모델용 데이터 생성 완료")
    print(f"Train: {train_count}장")
    print(f"Validation: {val_count}장")
    print(f"저장 위치: {LOCATOR_DATASET_DIR.resolve()}")
    if missing_roi:
        print(f"복부 ROI가 없어 제외된 이미지: {len(missing_roi)}장")
    if missing_navel:
        print(f"배꼽 좌표가 없어 제외된 이미지: {len(missing_navel)}장")
    if len(eligible_names) < 30:
        print("경고: 현재 데이터는 실행 확인용으로는 가능하지만 자동 검출 일반화에는 매우 적습니다.")
    print("다음 단계: python train_locator.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
