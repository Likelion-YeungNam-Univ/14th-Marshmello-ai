from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from davey_score import (
    QUADRANT_NAMES,
    analyze_quadrants,
    clean_binary_mask,
    collect_input_images,
    draw_result,
    predict_probability,
    shrink_roi_mask,
)
from locator_core import create_locator_model, predict_anatomy
from locator_settings import ABDOMEN_THRESHOLD, LOCATOR_CHECKPOINT_PATH
from settings import CHECKPOINT_DIR, NEW_IMAGES_DIR, OUTPUT_DIR
from unet_core import create_model


def save_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="복부 영역과 배꼽을 자동 검출하고 Davey's score를 계산합니다."
    )
    parser.add_argument("--input", type=Path, default=NEW_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "auto_davey_results")
    parser.add_argument("--stretch-checkpoint", type=Path, default=CHECKPOINT_DIR / "best_unet.pt")
    parser.add_argument("--locator-checkpoint", type=Path, default=LOCATOR_CHECKPOINT_PATH)
    parser.add_argument("--stretch-threshold", type=float, default=None)
    parser.add_argument("--abdomen-threshold", type=float, default=ABDOMEN_THRESHOLD)
    parser.add_argument("--roi-erode", type=int, default=3)
    parser.add_argument("--connect-pixels", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=25)
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument(
        "--minimum-navel-confidence",
        type=float,
        default=0.20,
        help="이 값보다 낮으면 결과 status를 review로 표시합니다.",
    )
    args = parser.parse_args()

    images = collect_input_images(args.input)
    if not images:
        raise RuntimeError(f"입력 이미지가 없습니다: {args.input}")
    if not args.stretch_checkpoint.exists():
        raise FileNotFoundError(
            f"임신선 U-Net 모델이 없습니다: {args.stretch_checkpoint}\n"
            "먼저 python train.py를 실행하세요."
        )
    if not args.locator_checkpoint.exists():
        raise FileNotFoundError(
            f"복부/배꼽 자동 모델이 없습니다: {args.locator_checkpoint}\n"
            "먼저 python train_locator.py를 실행하세요."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stretch_checkpoint = torch.load(
        args.stretch_checkpoint,
        map_location=device,
        weights_only=False,
    )
    stretch_image_size = int(stretch_checkpoint.get("image_size", 512))
    stretch_threshold = (
        float(args.stretch_threshold)
        if args.stretch_threshold is not None
        else float(stretch_checkpoint.get("threshold", 0.5))
    )
    stretch_encoder = str(stretch_checkpoint.get("encoder_name", "resnet34"))
    stretch_model = create_model(stretch_encoder, use_imagenet_weights=False).to(device)
    stretch_model.load_state_dict(stretch_checkpoint["model_state_dict"])
    stretch_model.eval()

    locator_checkpoint = torch.load(
        args.locator_checkpoint,
        map_location=device,
        weights_only=False,
    )
    locator_image_size = int(locator_checkpoint.get("image_size", 512))
    locator_encoder = str(locator_checkpoint.get("encoder_name", "resnet34"))
    locator_model = create_locator_model(
        locator_encoder,
        use_imagenet_weights=False,
    ).to(device)
    locator_model.load_state_dict(locator_checkpoint["model_state_dict"])
    locator_model.eval()

    dirs = {
        "stretch_masks": args.output / "stretch_masks",
        "abdomen_masks": args.output / "abdomen_masks",
        "roi_applied_masks": args.output / "roi_applied_masks",
        "cleaned_masks": args.output / "cleaned_masks",
        "visualizations": args.output / "visualizations",
        "json": args.output / "json",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"읽기 실패: {image_path}")
            continue

        anatomy = predict_anatomy(
            image,
            locator_model,
            device,
            image_size=locator_image_size,
            abdomen_threshold=args.abdomen_threshold,
        )

        abdomen_mask = shrink_roi_mask(
            anatomy.abdomen_mask,
            erode_pixels=max(0, args.roi_erode),
        )

        stretch_probability = predict_probability(
            image,
            stretch_model,
            device,
            stretch_image_size,
        )
        stretch_mask = (stretch_probability >= stretch_threshold).astype(np.uint8) * 255
        roi_applied = cv2.bitwise_and(stretch_mask, abdomen_mask)
        cleaned = clean_binary_mask(roi_applied, connect_pixels=args.connect_pixels)
        cleaned = cv2.bitwise_and(cleaned, abdomen_mask)

        navel_x = min(max(anatomy.navel_x, 1), image.shape[1] - 1)
        navel_y = min(max(anatomy.navel_y, 1), image.shape[0] - 1)

        quadrant_results = analyze_quadrants(
            cleaned,
            navel_x=navel_x,
            navel_y=navel_y,
            min_area=args.min_area,
            min_length=args.min_length,
        )
        total_score = sum(item.score for item in quadrant_results.values())

        status = "ok"
        warnings: list[str] = []
        if not anatomy.valid:
            status = "review"
            if anatomy.warning:
                warnings.append(anatomy.warning)
        if anatomy.navel_confidence < args.minimum_navel_confidence:
            status = "review"
            warnings.append(
                f"배꼽 신뢰도 낮음({anatomy.navel_confidence:.3f})"
            )

        cv2.imwrite(str(dirs["stretch_masks"] / f"{image_path.stem}.png"), stretch_mask)
        cv2.imwrite(str(dirs["abdomen_masks"] / f"{image_path.stem}.png"), abdomen_mask)
        cv2.imwrite(str(dirs["roi_applied_masks"] / f"{image_path.stem}.png"), roi_applied)
        cv2.imwrite(str(dirs["cleaned_masks"] / f"{image_path.stem}.png"), cleaned)

        visualization = draw_result(
            image,
            cleaned,
            abdomen_mask,
            navel_x,
            navel_y,
            quadrant_results,
            total_score,
        )
        status_text = f"AUTO STATUS: {status.upper()}"
        if warnings:
            status_text += " | " + "; ".join(warnings)
        y = min(82, visualization.shape[0] - 8)
        cv2.putText(
            visualization,
            status_text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            visualization,
            status_text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255) if status == "review" else (0, 200, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(
            str(dirs["visualizations"] / f"{image_path.stem}_auto_davey.jpg"),
            visualization,
        )

        detail = {
            "image_name": image_path.name,
            "status": status,
            "warning": "; ".join(warnings),
            "navel_x": navel_x,
            "navel_y": navel_y,
            "navel_confidence": anatomy.navel_confidence,
            "abdomen_confidence": anatomy.abdomen_confidence,
            "abdomen_polygon": [[x, y] for x, y in anatomy.abdomen_polygon],
            "left_upper": {
                "count": quadrant_results["left_upper"].count,
                "score": quadrant_results["left_upper"].score,
            },
            "right_upper": {
                "count": quadrant_results["right_upper"].count,
                "score": quadrant_results["right_upper"].score,
            },
            "left_lower": {
                "count": quadrant_results["left_lower"].count,
                "score": quadrant_results["left_lower"].score,
            },
            "right_lower": {
                "count": quadrant_results["right_lower"].count,
                "score": quadrant_results["right_lower"].score,
            },
            "total_score": total_score,
        }
        with (dirs["json"] / f"{image_path.stem}.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(detail, file, ensure_ascii=False, indent=2)

        row = {
            "image_name": image_path.name,
            "status": status,
            "warning": "; ".join(warnings),
            "navel_x": navel_x,
            "navel_y": navel_y,
            "navel_confidence": round(anatomy.navel_confidence, 6),
            "abdomen_confidence": round(anatomy.abdomen_confidence, 6),
            "left_upper_count": quadrant_results["left_upper"].count,
            "left_upper_score": quadrant_results["left_upper"].score,
            "right_upper_count": quadrant_results["right_upper"].count,
            "right_upper_score": quadrant_results["right_upper"].score,
            "left_lower_count": quadrant_results["left_lower"].count,
            "left_lower_score": quadrant_results["left_lower"].score,
            "right_lower_count": quadrant_results["right_lower"].count,
            "right_lower_score": quadrant_results["right_lower"].score,
            "total_score": total_score,
        }
        summary_rows.append(row)

        print("\n" + "=" * 64)
        print(f"이미지: {image_path.name}")
        print(f"자동 복부/배꼽 상태: {status}")
        if warnings:
            print(f"검토 사유: {'; '.join(warnings)}")
        print(f"배꼽: ({navel_x}, {navel_y}), confidence={anatomy.navel_confidence:.3f}")
        for name in ("left_upper", "right_upper", "left_lower", "right_lower"):
            result = quadrant_results[name]
            print(f"{QUADRANT_NAMES[name]}: {result.count}개 -> {result.score}점")
        print(f"Davey's score: {total_score}/8")
        print("=" * 64)

    save_summary(args.output / "auto_davey_scores.csv", summary_rows)
    print(f"\n완료: {args.output.resolve()}")
    print("status가 review인 결과는 시각화 파일을 반드시 확인하세요.")


if __name__ == "__main__":
    main()
