from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from davey_score import (
    collect_input_images,
    load_abdomen_rois,
    load_navel_points,
    save_abdomen_rois,
    save_navel_points,
)
from label_ui import (
    select_abdomen_polygon_interactively,
    select_navel_interactively,
)
from settings import NEW_IMAGES_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="자동 위치 모델 학습용 복부 외곽과 배꼽 좌표를 수동으로 한 번 저장합니다."
    )
    parser.add_argument("--input", type=Path, default=NEW_IMAGES_DIR)
    parser.add_argument("--abdomen-roi-json", type=Path, default=Path("abdomen_rois.json"))
    parser.add_argument("--navel-csv", type=Path, default=Path("navel_points.csv"))
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    images = collect_input_images(args.input)
    if not images:
        raise RuntimeError(f"이미지가 없습니다: {args.input}")

    rois = load_abdomen_rois(args.abdomen_roi_json)
    navels = load_navel_points(args.navel_csv)

    for image_path in images:
        if not args.redo and image_path.name in rois and image_path.name in navels:
            print(f"이미 라벨 있음, 건너뜀: {image_path.name}")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"읽기 실패: {image_path}")
            continue

        print(f"\n라벨링: {image_path.name}")
        roi = select_abdomen_polygon_interactively(image, image_path.name)
        navel = select_navel_interactively(image, image_path.name)
        rois[image_path.name] = roi
        navels[image_path.name] = navel
        save_abdomen_rois(args.abdomen_roi_json, rois)
        save_navel_points(args.navel_csv, navels)

    print("\n라벨 저장 완료")
    print(f"복부 ROI: {args.abdomen_roi_json.resolve()}")
    print(f"배꼽 좌표: {args.navel_csv.resolve()}")


if __name__ == "__main__":
    main()
