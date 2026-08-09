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
from input_validator import InputStatus, draw_validation_overlay, validate_input_image
from locator_core import create_locator_model, predict_anatomy
from locator_settings import (
    ABDOMEN_THRESHOLD,
    LOCATOR_CHECKPOINT_PATH,
    LOCATOR_VERSION,
    MIN_ABDOMEN_CORE_WIDTH_RATIO_REVIEW,
    MIN_ABDOMEN_SOLIDITY_REVIEW,
    MIN_NAVEL_BOUNDARY_MARGIN_RATIO_REVIEW,
    NAVEL_TARGET_TYPE,
)
from semantic_validator import (
    SemanticInputValidator,
    SemanticStatus,
    SemanticValidationResult,
    draw_semantic_overlay,
)
from settings import CHECKPOINT_DIR, NEW_IMAGES_DIR, OUTPUT_DIR
from unet_core import create_model


SUMMARY_FIELDS = [
    "image_name",
    "status",
    "score_calculated",
    "reason_codes",
    "warning",
    "semantic_status",
    "semantic_target_probability",
    "semantic_positive_similarity",
    "semantic_negative_similarity",
    "semantic_top_negative_prompt",
    "navel_x",
    "navel_y",
    "navel_confidence",
    "abdomen_confidence",
    "abdomen_area_ratio",
    "abdomen_solidity",
    "abdomen_extent",
    "abdomen_core_width_ratio",
    "abdomen_component_count",
    "navel_boundary_margin_ratio",
    "navel_heatmap_concentration",
    "navel_heatmap_spread_ratio",
    "navel_candidate_area_ratio",
    "navel_candidate_component_count",
    "brightness_mean",
    "contrast_std",
    "left_upper_count",
    "left_upper_score",
    "right_upper_count",
    "right_upper_score",
    "left_lower_count",
    "left_lower_score",
    "right_lower_count",
    "right_lower_score",
    "total_score",
]


def save_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, detail: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(detail, file, ensure_ascii=False, indent=2)


def empty_quadrant_detail() -> dict[str, dict[str, None]]:
    return {
        "left_upper": {"count": None, "score": None},
        "right_upper": {"count": None, "score": None},
        "left_lower": {"count": None, "score": None},
        "right_lower": {"count": None, "score": None},
    }


def empty_quadrant_row() -> dict[str, None]:
    return {
        "left_upper_count": None,
        "left_upper_score": None,
        "right_upper_count": None,
        "right_upper_score": None,
        "left_lower_count": None,
        "left_lower_score": None,
        "right_lower_count": None,
        "right_lower_score": None,
    }


def semantic_output_fields(
    result: SemanticValidationResult | None,
) -> dict[str, object]:
    if result is None:
        return {
            "semantic_status": "skipped",
            "semantic_target_probability": None,
            "semantic_positive_similarity": None,
            "semantic_negative_similarity": None,
            "semantic_top_negative_prompt": "",
        }
    return {
        "semantic_status": result.status.value,
        "semantic_target_probability": round(result.target_probability, 6),
        "semantic_positive_similarity": round(result.positive_similarity, 6),
        "semantic_negative_similarity": round(result.negative_similarity, 6),
        "semantic_top_negative_prompt": result.top_negative_prompt,
    }


def validation_summary_fields(validation) -> dict[str, object]:
    def rounded(value, digits=6):
        return None if value is None else round(value, digits)

    return {
        "abdomen_area_ratio": round(validation.abdomen_area_ratio, 6),
        "abdomen_solidity": rounded(validation.abdomen_solidity),
        "abdomen_extent": rounded(validation.abdomen_extent),
        "abdomen_core_width_ratio": rounded(validation.abdomen_core_width_ratio),
        "abdomen_component_count": validation.abdomen_component_count,
        "navel_boundary_margin_ratio": rounded(validation.navel_boundary_margin_ratio),
        "navel_heatmap_concentration": rounded(validation.navel_heatmap_concentration),
        "navel_heatmap_spread_ratio": rounded(validation.navel_heatmap_spread_ratio),
        "navel_candidate_area_ratio": rounded(validation.navel_candidate_area_ratio),
        "navel_candidate_component_count": validation.navel_candidate_component_count,
        "brightness_mean": round(validation.brightness_mean, 3),
        "contrast_std": round(validation.contrast_std, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "복부/배꼽 입력 유효성을 먼저 검사한 뒤, 유효한 이미지에 대해서만 "
            "Davey's score를 계산합니다."
        )
    )
    parser.add_argument("--input", type=Path, default=NEW_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "auto_davey_results")
    parser.add_argument(
        "--stretch-checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "best_unet.pt",
    )
    parser.add_argument(
        "--locator-checkpoint",
        type=Path,
        default=LOCATOR_CHECKPOINT_PATH,
    )
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
        help=(
            "배꼽 confidence가 이 값보다 낮으면 review 처리합니다. "
            "review는 기본적으로 점수를 계산하지 않습니다."
        ),
    )
    parser.add_argument(
        "--minimum-abdomen-confidence",
        type=float,
        default=0.55,
        help=(
            "복부 confidence가 이 값보다 낮으면 review 처리합니다. "
            "실제 배포 전 validation set으로 보정하는 것을 권장합니다."
        ),
    )
    parser.add_argument("--minimum-image-width", type=int, default=128)
    parser.add_argument("--minimum-image-height", type=int, default=128)
    parser.add_argument("--minimum-brightness", type=float, default=18.0)
    parser.add_argument("--maximum-brightness", type=float, default=242.0)
    parser.add_argument("--minimum-contrast-std", type=float, default=8.0)
    parser.add_argument("--navel-edge-margin-ratio", type=float, default=0.04)
    parser.add_argument(
        "--minimum-abdomen-solidity-review",
        type=float,
        default=MIN_ABDOMEN_SOLIDITY_REVIEW,
        help="복부 mask solidity가 이 값보다 낮으면 review 처리합니다.",
    )
    parser.add_argument(
        "--minimum-abdomen-core-width-ratio-review",
        type=float,
        default=MIN_ABDOMEN_CORE_WIDTH_RATIO_REVIEW,
        help="복부 mask 중앙 폭 비율이 이 값보다 낮으면 review 처리합니다.",
    )
    parser.add_argument(
        "--minimum-navel-boundary-margin-ratio-review",
        type=float,
        default=MIN_NAVEL_BOUNDARY_MARGIN_RATIO_REVIEW,
        help="배꼽과 복부 외곽선 사이 여유가 이 값보다 작으면 review 처리합니다.",
    )
    parser.add_argument(
        "--allow-legacy-locator",
        action="store_true",
        help=(
            "구형 원형-mask locator checkpoint를 테스트 목적으로 허용합니다. "
            "운영에서는 locator v2 재학습 checkpoint 사용을 권장합니다."
        ),
    )
    parser.add_argument(
        "--allow-review-scoring",
        action="store_true",
        help=(
            "review 이미지에도 Davey's score 계산을 허용합니다. "
            "운영 환경에서는 기본값(False)을 권장합니다."
        ),
    )
    parser.add_argument(
        "--skip-semantic-validation",
        action="store_true",
        help=(
            "OpenCLIP 기반 사람 복부 사진 검증을 끕니다. "
            "이 옵션을 사용하면 동물/사물 같은 OOD 입력 방어력이 크게 낮아집니다."
        ),
    )
    parser.add_argument("--semantic-model", type=str, default="ViT-B-32")
    parser.add_argument(
        "--semantic-pretrained",
        type=str,
        default="laion2b_s34b_b79k",
    )
    parser.add_argument(
        "--semantic-reject-probability",
        type=float,
        default=0.40,
        help="이 값보다 낮으면 NOT_TARGET_IMAGE로 즉시 거부합니다.",
    )
    parser.add_argument(
        "--semantic-valid-probability",
        type=float,
        default=0.65,
        help="이 값 이상이어야 semantic gate를 valid로 통과합니다.",
    )
    parser.add_argument(
        "--semantic-comparison-scale",
        type=float,
        default=30.0,
        help="positive/negative CLIP similarity 차이를 확률로 바꿀 때 사용하는 scale입니다.",
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

    semantic_validator = None
    if not args.skip_semantic_validation:
        print(
            "semantic input validator 로딩 중... "
            f"model={args.semantic_model}, pretrained={args.semantic_pretrained}"
        )
        semantic_validator = SemanticInputValidator(
            device=device,
            model_name=args.semantic_model,
            pretrained=args.semantic_pretrained,
            reject_probability=args.semantic_reject_probability,
            valid_probability=args.semantic_valid_probability,
            comparison_scale=args.semantic_comparison_scale,
        )

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
    locator_is_v2 = (
        int(locator_checkpoint.get("locator_version", 0)) >= LOCATOR_VERSION
        and str(locator_checkpoint.get("navel_target_type", "")) == NAVEL_TARGET_TYPE
    )
    if not locator_is_v2:
        message = (
            "현재 locator checkpoint는 Gaussian heatmap 기반 locator v2가 아닙니다.\n"
            "다음을 실행해 locator를 재학습하세요:\n"
            "  python prepare_locator_dataset.py --overwrite\n"
            "  python train_locator.py"
        )
        if not args.allow_legacy_locator:
            raise RuntimeError(message)
        print("WARNING: " + message.replace("\n", " | "))

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
        "input_validation": args.output / "input_validation",
        "semantic_validation": args.output / "semantic_validation",
        "json": args.output / "json",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            warning = "이미지를 읽을 수 없습니다. 파일 손상 또는 지원하지 않는 형식일 수 있습니다."
            semantic_fields = {
                "semantic_status": InputStatus.INVALID.value,
                "semantic_target_probability": None,
                "semantic_positive_similarity": None,
                "semantic_negative_similarity": None,
                "semantic_top_negative_prompt": "",
            }
            detail = {
                "image_name": image_path.name,
                "status": InputStatus.INVALID.value,
                "score_calculated": False,
                "reason_codes": ["UNREADABLE_IMAGE"],
                "warning": warning,
                **semantic_fields,
                "navel_x": None,
                "navel_y": None,
                "navel_confidence": None,
                "abdomen_confidence": None,
                "abdomen_polygon": [],
                "validation_metrics": {
                    "abdomen_area_ratio": None,
                    "brightness_mean": None,
                    "contrast_std": None,
                },
                **empty_quadrant_detail(),
                "total_score": None,
            }
            save_json(dirs["json"] / f"{image_path.stem}.json", detail)
            summary_rows.append(
                {
                    "image_name": image_path.name,
                    "status": InputStatus.INVALID.value,
                    "score_calculated": False,
                    "reason_codes": "UNREADABLE_IMAGE",
                    "warning": warning,
                    **semantic_fields,
                    "navel_x": None,
                    "navel_y": None,
                    "navel_confidence": None,
                    "abdomen_confidence": None,
                    "abdomen_area_ratio": None,
                    "brightness_mean": None,
                    "contrast_std": None,
                    **empty_quadrant_row(),
                    "total_score": None,
                }
            )
            print(f"{image_path.name}: INVALID INPUT - {warning}")
            continue

        semantic_result: SemanticValidationResult | None = None
        if semantic_validator is not None:
            semantic_result = semantic_validator.predict(image)
            semantic_visualization = draw_semantic_overlay(image, semantic_result)
            cv2.imwrite(
                str(
                    dirs["semantic_validation"]
                    / f"{image_path.stem}_semantic_validation.jpg"
                ),
                semantic_visualization,
            )

            # The semantic gate is intentionally before the task-specific locator.
            # If this is clearly not a human-abdomen image, do not let the locator
            # hallucinate an abdomen/navel and do not calculate Davey's score.
            if semantic_result.status == SemanticStatus.INVALID:
                semantic_fields = semantic_output_fields(semantic_result)
                detail = {
                    "image_name": image_path.name,
                    "status": InputStatus.INVALID.value,
                    "score_calculated": False,
                    "reason_codes": [semantic_result.reason_code],
                    "warning": semantic_result.warning,
                    **semantic_fields,
                    "semantic_top_positive_prompt": semantic_result.top_positive_prompt,
                    "navel_x": None,
                    "navel_y": None,
                    "navel_confidence": None,
                    "abdomen_confidence": None,
                    "abdomen_polygon": [],
                    "validation_metrics": {
                        "abdomen_area_ratio": None,
                        "brightness_mean": None,
                        "contrast_std": None,
                    },
                    **empty_quadrant_detail(),
                    "total_score": None,
                }
                save_json(dirs["json"] / f"{image_path.stem}.json", detail)
                summary_rows.append(
                    {
                        "image_name": image_path.name,
                        "status": InputStatus.INVALID.value,
                        "score_calculated": False,
                        "reason_codes": semantic_result.reason_code,
                        "warning": semantic_result.warning,
                        **semantic_fields,
                        "navel_x": None,
                        "navel_y": None,
                        "navel_confidence": None,
                        "abdomen_confidence": None,
                        "abdomen_area_ratio": None,
                        "brightness_mean": None,
                        "contrast_std": None,
                        **empty_quadrant_row(),
                        "total_score": None,
                    }
                )
                print("\n" + "=" * 64)
                print(f"이미지: {image_path.name}")
                print("입력 검증 상태: invalid_input")
                print("점수 계산: 중단 (semantic gate)")
                print(f"사유: {semantic_result.warning}")
                print(
                    "semantic target probability: "
                    f"{semantic_result.target_probability:.3f}"
                )
                print(f"가장 가까운 비대상 prompt: {semantic_result.top_negative_prompt}")
                print("Davey's score: N/A")
                print("=" * 64)
                continue

        anatomy = predict_anatomy(
            image,
            locator_model,
            device,
            image_size=locator_image_size,
            abdomen_threshold=args.abdomen_threshold,
        )

        validation = validate_input_image(
            image,
            anatomy,
            minimum_navel_confidence=args.minimum_navel_confidence,
            minimum_abdomen_confidence=args.minimum_abdomen_confidence,
            minimum_width=max(1, args.minimum_image_width),
            minimum_height=max(1, args.minimum_image_height),
            minimum_brightness=args.minimum_brightness,
            maximum_brightness=args.maximum_brightness,
            minimum_contrast_std=max(0.0, args.minimum_contrast_std),
            navel_edge_margin_ratio=max(0.0, args.navel_edge_margin_ratio),
            minimum_abdomen_solidity_review=max(0.0, args.minimum_abdomen_solidity_review),
            minimum_abdomen_core_width_ratio_review=max(
                0.0, args.minimum_abdomen_core_width_ratio_review
            ),
            minimum_navel_boundary_margin_ratio_review=max(
                0.0, args.minimum_navel_boundary_margin_ratio_review
            ),
        )

        cv2.imwrite(
            str(dirs["abdomen_masks"] / f"{image_path.stem}.png"),
            anatomy.abdomen_mask,
        )
        validation_visualization = draw_validation_overlay(image, anatomy, validation)
        cv2.imwrite(
            str(dirs["input_validation"] / f"{image_path.stem}_input_validation.jpg"),
            validation_visualization,
        )

        reason_codes = list(validation.reason_codes)
        warnings = [issue.message for issue in validation.issues]
        combined_status = validation.status

        if (
            semantic_result is not None
            and semantic_result.status == SemanticStatus.REVIEW
        ):
            if semantic_result.reason_code:
                reason_codes.insert(0, semantic_result.reason_code)
            if semantic_result.warning:
                warnings.insert(0, semantic_result.warning)
            if combined_status == InputStatus.VALID:
                combined_status = InputStatus.REVIEW

        should_score = combined_status == InputStatus.VALID or (
            combined_status == InputStatus.REVIEW and args.allow_review_scoring
        )
        semantic_fields = semantic_output_fields(semantic_result)

        if not should_score:
            navel_x = (
                int(anatomy.navel_x)
                if 0 <= int(anatomy.navel_x) < image.shape[1]
                else None
            )
            navel_y = (
                int(anatomy.navel_y)
                if 0 <= int(anatomy.navel_y) < image.shape[0]
                else None
            )
            warning_text = "; ".join(warnings)
            detail = {
                "image_name": image_path.name,
                "status": combined_status.value,
                "score_calculated": False,
                "reason_codes": reason_codes,
                "warning": warning_text,
                **semantic_fields,
                "semantic_top_positive_prompt": (
                    semantic_result.top_positive_prompt if semantic_result else ""
                ),
                "navel_x": navel_x,
                "navel_y": navel_y,
                "navel_confidence": anatomy.navel_confidence,
                "abdomen_confidence": anatomy.abdomen_confidence,
                "abdomen_polygon": [[x, y] for x, y in anatomy.abdomen_polygon],
                "validation_metrics": validation.metrics,
                **empty_quadrant_detail(),
                "total_score": None,
            }
            save_json(dirs["json"] / f"{image_path.stem}.json", detail)
            summary_rows.append(
                {
                    "image_name": image_path.name,
                    "status": combined_status.value,
                    "score_calculated": False,
                    "reason_codes": "|".join(reason_codes),
                    "warning": warning_text,
                    **semantic_fields,
                    "navel_x": navel_x,
                    "navel_y": navel_y,
                    "navel_confidence": round(anatomy.navel_confidence, 6),
                    "abdomen_confidence": round(anatomy.abdomen_confidence, 6),
                    **validation_summary_fields(validation),
                    **empty_quadrant_row(),
                    "total_score": None,
                }
            )

            print("\n" + "=" * 64)
            print(f"이미지: {image_path.name}")
            print(f"입력 검증 상태: {combined_status.value}")
            if semantic_result is not None:
                print(
                    "semantic target probability: "
                    f"{semantic_result.target_probability:.3f}"
                )
            print("점수 계산: 중단")
            if warning_text:
                print(f"사유: {warning_text}")
            print("Davey's score: N/A")
            print("=" * 64)
            continue

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

        navel_x = min(max(int(anatomy.navel_x), 1), image.shape[1] - 1)
        navel_y = min(max(int(anatomy.navel_y), 1), image.shape[0] - 1)

        quadrant_results = analyze_quadrants(
            cleaned,
            navel_x=navel_x,
            navel_y=navel_y,
            min_area=args.min_area,
            min_length=args.min_length,
        )
        total_score = sum(item.score for item in quadrant_results.values())

        status = (
            "ok"
            if combined_status == InputStatus.VALID
            else InputStatus.REVIEW.value
        )
        warning_text = "; ".join(warnings)

        cv2.imwrite(
            str(dirs["stretch_masks"] / f"{image_path.stem}.png"),
            stretch_mask,
        )
        cv2.imwrite(
            str(dirs["roi_applied_masks"] / f"{image_path.stem}.png"),
            roi_applied,
        )
        cv2.imwrite(
            str(dirs["cleaned_masks"] / f"{image_path.stem}.png"),
            cleaned,
        )

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
        if warning_text:
            status_text += " | " + warning_text
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
            (0, 0, 255) if status == InputStatus.REVIEW.value else (0, 200, 0),
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
            "score_calculated": True,
            "reason_codes": reason_codes,
            "warning": warning_text,
            **semantic_fields,
            "semantic_top_positive_prompt": (
                semantic_result.top_positive_prompt if semantic_result else ""
            ),
            "navel_x": navel_x,
            "navel_y": navel_y,
            "navel_confidence": anatomy.navel_confidence,
            "abdomen_confidence": anatomy.abdomen_confidence,
            "abdomen_polygon": [[x, y] for x, y in anatomy.abdomen_polygon],
            "validation_metrics": validation.metrics,
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
        save_json(dirs["json"] / f"{image_path.stem}.json", detail)

        summary_rows.append(
            {
                "image_name": image_path.name,
                "status": status,
                "score_calculated": True,
                "reason_codes": "|".join(reason_codes),
                "warning": warning_text,
                **semantic_fields,
                "navel_x": navel_x,
                "navel_y": navel_y,
                "navel_confidence": round(anatomy.navel_confidence, 6),
                "abdomen_confidence": round(anatomy.abdomen_confidence, 6),
                **validation_summary_fields(validation),
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
        )

        print("\n" + "=" * 64)
        print(f"이미지: {image_path.name}")
        print(f"입력 검증 상태: {combined_status.value}")
        if semantic_result is not None:
            print(
                "semantic target probability: "
                f"{semantic_result.target_probability:.3f}"
            )
        print(f"자동 복부/배꼽 상태: {status}")
        if warning_text:
            print(f"검토 사유: {warning_text}")
        print(
            f"배꼽: ({navel_x}, {navel_y}), "
            f"confidence={anatomy.navel_confidence:.3f}"
        )
        for name in ("left_upper", "right_upper", "left_lower", "right_lower"):
            result = quadrant_results[name]
            print(f"{QUADRANT_NAMES[name]}: {result.count}개 -> {result.score}점")
        print(f"Davey's score: {total_score}/8")
        print("=" * 64)

    save_summary(args.output / "auto_davey_scores.csv", summary_rows)
    print(f"\n완료: {args.output.resolve()}")
    print(
        "invalid_input/review는 기본적으로 Davey's score를 계산하지 않습니다. "
        "review 점수가 꼭 필요할 때만 --allow-review-scoring을 사용하세요."
    )


if __name__ == "__main__":
    main()
