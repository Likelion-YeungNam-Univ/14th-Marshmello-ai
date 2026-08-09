from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from locator_core import create_locator_model, list_images, predict_anatomy
from locator_settings import (
    ABDOMEN_THRESHOLD,
    LOCATOR_CHECKPOINT_PATH,
    LOCATOR_OUTPUT_DIR,
)
from settings import NEW_IMAGES_DIR


def create_visualization(
    image: np.ndarray,
    abdomen_mask: np.ndarray,
    polygon: list[tuple[int, int]],
    navel_x: int,
    navel_y: int,
    valid: bool,
    warning: str,
) -> np.ndarray:
    result = image.copy()
    overlay = np.zeros_like(result)
    overlay[:, :, 1] = 255
    selected = abdomen_mask > 0
    result[selected] = (0.75 * result[selected] + 0.25 * overlay[selected]).astype(np.uint8)

    if len(polygon) >= 3:
        points = np.asarray(polygon, dtype=np.int32)
        cv2.polylines(result, [points], True, (255, 255, 0), 3)

    cv2.circle(result, (navel_x, navel_y), 10, (255, 0, 255), -1)
    cv2.line(result, (navel_x - 18, navel_y), (navel_x + 18, navel_y), (0, 255, 255), 2)
    cv2.line(result, (navel_x, navel_y - 18), (navel_x, navel_y + 18), (0, 255, 255), 2)

    status = "OK" if valid else "REVIEW"
    cv2.rectangle(result, (0, 0), (result.shape[1] - 1, 58), (0, 0, 0), -1)
    cv2.putText(
        result,
        f"Auto anatomy: {status} {warning}",
        (12, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="복부 영역과 배꼽 좌표를 자동 예측합니다.")
    parser.add_argument("--input", type=Path, default=NEW_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=LOCATOR_OUTPUT_DIR / "predictions")
    parser.add_argument("--checkpoint", type=Path, default=LOCATOR_CHECKPOINT_PATH)
    parser.add_argument("--abdomen-threshold", type=float, default=ABDOMEN_THRESHOLD)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"자동 위치 모델이 없습니다: {args.checkpoint}\n"
            "먼저 python train_locator.py를 실행하세요."
        )

    images = [args.input] if args.input.is_file() else list_images(args.input)
    if not images:
        raise RuntimeError(f"입력 이미지가 없습니다: {args.input}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 512))
    encoder_name = str(checkpoint.get("encoder_name", "resnet34"))

    model = create_locator_model(encoder_name, use_imagenet_weights=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    abdomen_dir = args.output / "abdomen_masks"
    heatmap_dir = args.output / "navel_heatmaps"
    visualization_dir = args.output / "visualizations"
    for directory in (abdomen_dir, heatmap_dir, visualization_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    roi_json: dict[str, list[list[int]]] = {}

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"읽기 실패: {image_path}")
            continue

        prediction = predict_anatomy(
            image,
            model,
            device,
            image_size=image_size,
            abdomen_threshold=args.abdomen_threshold,
        )

        cv2.imwrite(str(abdomen_dir / f"{image_path.stem}.png"), prediction.abdomen_mask)
        heatmap = np.clip(prediction.navel_probability * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(heatmap_dir / f"{image_path.stem}.png"), heatmap)
        visualization = create_visualization(
            image,
            prediction.abdomen_mask,
            prediction.abdomen_polygon,
            prediction.navel_x,
            prediction.navel_y,
            prediction.valid,
            prediction.warning,
        )
        cv2.imwrite(
            str(visualization_dir / f"{image_path.stem}_anatomy.jpg"),
            visualization,
        )

        roi_json[image_path.name] = [
            [int(x), int(y)] for x, y in prediction.abdomen_polygon
        ]
        rows.append(
            {
                "image_name": image_path.name,
                "navel_x": prediction.navel_x,
                "navel_y": prediction.navel_y,
                "navel_confidence": round(prediction.navel_confidence, 6),
                "abdomen_confidence": round(prediction.abdomen_confidence, 6),
                "valid": prediction.valid,
                "warning": prediction.warning,
            }
        )
        print(
            f"{image_path.name}: navel=({prediction.navel_x}, {prediction.navel_y}), "
            f"valid={prediction.valid}, warning={prediction.warning}"
        )

    csv_path = args.output / "auto_navel_points.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output / "auto_abdomen_rois.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(roi_json, file, ensure_ascii=False, indent=2)

    print(f"\n완료: {args.output.resolve()}")
    print("visualizations 폴더에서 복부 외곽과 배꼽 위치를 확인하세요.")


if __name__ == "__main__":
    main()
