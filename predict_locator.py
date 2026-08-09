from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from input_validator import draw_validation_overlay, validate_input_image
from locator_core import create_locator_model, list_images, predict_anatomy
from locator_settings import (
    ABDOMEN_THRESHOLD,
    LOCATOR_CHECKPOINT_PATH,
    LOCATOR_OUTPUT_DIR,
    LOCATOR_VERSION,
    NAVEL_TARGET_TYPE,
)
from settings import NEW_IMAGES_DIR


def checkpoint_is_locator_v2(checkpoint: dict[str, object]) -> bool:
    return (
        int(checkpoint.get("locator_version", 0)) >= LOCATOR_VERSION
        and str(checkpoint.get("navel_target_type", "")) == NAVEL_TARGET_TYPE
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="복부 영역과 배꼽 좌표를 locator v2로 예측합니다.")
    parser.add_argument("--input", type=Path, default=NEW_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=LOCATOR_OUTPUT_DIR / "predictions")
    parser.add_argument("--checkpoint", type=Path, default=LOCATOR_CHECKPOINT_PATH)
    parser.add_argument("--abdomen-threshold", type=float, default=ABDOMEN_THRESHOLD)
    parser.add_argument(
        "--allow-legacy-locator",
        action="store_true",
        help="구형 원형-mask locator checkpoint를 테스트 목적으로 허용합니다.",
    )
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
    if not checkpoint_is_locator_v2(checkpoint):
        message = (
            "현재 checkpoint는 locator v2 Gaussian heatmap 학습 모델이 아닙니다.\n"
            "다음을 실행해 dataset을 다시 만들고 locator를 재학습하세요:\n"
            "  python prepare_locator_dataset.py --overwrite\n"
            "  python train_locator.py"
        )
        if not args.allow_legacy_locator:
            raise RuntimeError(message)
        print("WARNING: " + message.replace("\n", " | "))

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
        validation = validate_input_image(image, prediction)

        cv2.imwrite(str(abdomen_dir / f"{image_path.stem}.png"), prediction.abdomen_mask)
        heatmap = np.clip(prediction.navel_probability * 255.0, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(str(heatmap_dir / f"{image_path.stem}.png"), heatmap_color)

        visualization = draw_validation_overlay(image, prediction, validation)
        cv2.imwrite(
            str(visualization_dir / f"{image_path.stem}_anatomy.jpg"),
            visualization,
        )

        roi_json[image_path.name] = [[int(x), int(y)] for x, y in prediction.abdomen_polygon]
        rows.append(
            {
                "image_name": image_path.name,
                "status": validation.status.value,
                "reason_codes": "|".join(validation.reason_codes),
                "navel_x": prediction.navel_x,
                "navel_y": prediction.navel_y,
                "navel_confidence": round(prediction.navel_confidence, 6),
                "abdomen_confidence": round(prediction.abdomen_confidence, 6),
                "abdomen_solidity": round(prediction.abdomen_solidity, 6),
                "abdomen_extent": round(prediction.abdomen_extent, 6),
                "abdomen_core_width_ratio": round(prediction.abdomen_core_width_ratio, 6),
                "abdomen_component_count": prediction.abdomen_component_count,
                "navel_boundary_margin_ratio": round(
                    prediction.navel_boundary_margin_ratio, 6
                ),
                "navel_heatmap_concentration": round(
                    prediction.navel_heatmap_concentration, 6
                ),
                "navel_heatmap_spread_ratio": round(
                    prediction.navel_heatmap_spread_ratio, 6
                ),
                "navel_candidate_area_ratio": round(
                    prediction.navel_candidate_area_ratio, 6
                ),
                "valid": prediction.valid,
                "warning": validation.warning,
            }
        )
        print(
            f"{image_path.name}: status={validation.status.value}, "
            f"navel=({prediction.navel_x}, {prediction.navel_y}), "
            f"solidity={prediction.abdomen_solidity:.3f}, "
            f"navel_margin={prediction.navel_boundary_margin_ratio:.3f}, "
            f"reason={','.join(validation.reason_codes) or '-'}"
        )

    if not rows:
        raise RuntimeError("처리 가능한 이미지가 없습니다.")

    csv_path = args.output / "auto_navel_points.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output / "auto_abdomen_rois.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(roi_json, file, ensure_ascii=False, indent=2)

    print(f"\n완료: {args.output.resolve()}")
    print("visualizations/ 및 navel_heatmaps/ 폴더에서 위치와 heatmap을 확인하세요.")


if __name__ == "__main__":
    main()
