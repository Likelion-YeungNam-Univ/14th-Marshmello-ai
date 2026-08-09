from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import cv2
import numpy as np

from locator_settings import (
    MAX_NAVEL_CANDIDATE_AREA_RATIO,
    MAX_NAVEL_HEATMAP_SPREAD_RATIO,
    MIN_ABDOMEN_CORE_WIDTH_RATIO_HARD,
    MIN_ABDOMEN_CORE_WIDTH_RATIO_REVIEW,
    MIN_ABDOMEN_EXTENT_HARD,
    MIN_ABDOMEN_EXTENT_REVIEW,
    MIN_ABDOMEN_SOLIDITY_HARD,
    MIN_ABDOMEN_SOLIDITY_REVIEW,
    MIN_NAVEL_BOUNDARY_MARGIN_RATIO_HARD,
    MIN_NAVEL_BOUNDARY_MARGIN_RATIO_REVIEW,
    MIN_NAVEL_HEATMAP_CONCENTRATION,
)


class AnatomyLike(Protocol):
    abdomen_mask: np.ndarray
    abdomen_polygon: list[tuple[int, int]]
    navel_x: int
    navel_y: int
    navel_confidence: float
    abdomen_confidence: float
    valid: bool
    warning: str


class InputStatus(str, Enum):
    VALID = "valid"
    REVIEW = "review"
    INVALID = "invalid_input"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    hard_reject: bool


@dataclass(frozen=True)
class InputValidationResult:
    status: InputStatus
    issues: tuple[ValidationIssue, ...]
    abdomen_area_ratio: float
    brightness_mean: float
    contrast_std: float
    abdomen_solidity: float | None = None
    abdomen_extent: float | None = None
    abdomen_core_width_ratio: float | None = None
    abdomen_component_count: int | None = None
    navel_boundary_margin_ratio: float | None = None
    navel_heatmap_concentration: float | None = None
    navel_heatmap_spread_ratio: float | None = None
    navel_candidate_area_ratio: float | None = None
    navel_candidate_component_count: int | None = None

    @property
    def can_score(self) -> bool:
        return self.status == InputStatus.VALID

    @property
    def warning(self) -> str:
        return "; ".join(issue.message for issue in self.issues)

    @property
    def reason_codes(self) -> list[str]:
        return [issue.code for issue in self.issues]

    @property
    def metrics(self) -> dict[str, float | int | None]:
        return {
            "abdomen_area_ratio": self.abdomen_area_ratio,
            "brightness_mean": self.brightness_mean,
            "contrast_std": self.contrast_std,
            "abdomen_solidity": self.abdomen_solidity,
            "abdomen_extent": self.abdomen_extent,
            "abdomen_core_width_ratio": self.abdomen_core_width_ratio,
            "abdomen_component_count": self.abdomen_component_count,
            "navel_boundary_margin_ratio": self.navel_boundary_margin_ratio,
            "navel_heatmap_concentration": self.navel_heatmap_concentration,
            "navel_heatmap_spread_ratio": self.navel_heatmap_spread_ratio,
            "navel_candidate_area_ratio": self.navel_candidate_area_ratio,
            "navel_candidate_component_count": self.navel_candidate_component_count,
        }


def _add_issue(
    issues: list[ValidationIssue],
    seen_codes: set[str],
    *,
    code: str,
    message: str,
    hard_reject: bool,
) -> None:
    if code in seen_codes:
        return
    seen_codes.add(code)
    issues.append(
        ValidationIssue(
            code=code,
            message=message,
            hard_reject=hard_reject,
        )
    )


def _image_statistics(image_bgr: np.ndarray) -> tuple[float, float]:
    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def _optional_float(anatomy: AnatomyLike, name: str) -> float | None:
    value = getattr(anatomy, name, None)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _optional_int(anatomy: AnatomyLike, name: str) -> int | None:
    value = getattr(anatomy, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_input_image(
    image_bgr: np.ndarray,
    anatomy: AnatomyLike,
    *,
    minimum_navel_confidence: float = 0.20,
    minimum_abdomen_confidence: float = 0.55,
    minimum_width: int = 128,
    minimum_height: int = 128,
    minimum_brightness: float = 18.0,
    maximum_brightness: float = 242.0,
    minimum_contrast_std: float = 8.0,
    navel_edge_margin_ratio: float = 0.04,
    minimum_abdomen_solidity_review: float = MIN_ABDOMEN_SOLIDITY_REVIEW,
    minimum_abdomen_solidity_hard: float = MIN_ABDOMEN_SOLIDITY_HARD,
    minimum_abdomen_extent_review: float = MIN_ABDOMEN_EXTENT_REVIEW,
    minimum_abdomen_extent_hard: float = MIN_ABDOMEN_EXTENT_HARD,
    minimum_abdomen_core_width_ratio_review: float = MIN_ABDOMEN_CORE_WIDTH_RATIO_REVIEW,
    minimum_abdomen_core_width_ratio_hard: float = MIN_ABDOMEN_CORE_WIDTH_RATIO_HARD,
    minimum_navel_boundary_margin_ratio_review: float = MIN_NAVEL_BOUNDARY_MARGIN_RATIO_REVIEW,
    minimum_navel_boundary_margin_ratio_hard: float = MIN_NAVEL_BOUNDARY_MARGIN_RATIO_HARD,
    minimum_navel_heatmap_concentration: float = MIN_NAVEL_HEATMAP_CONCENTRATION,
    maximum_navel_heatmap_spread_ratio: float = MAX_NAVEL_HEATMAP_SPREAD_RATIO,
    maximum_navel_candidate_area_ratio: float = MAX_NAVEL_CANDIDATE_AREA_RATIO,
) -> InputValidationResult:
    """Validate image quality, anatomy geometry and navel heatmap stability.

    Hard failures become ``invalid_input``. Soft failures become ``review`` and are
    not scored by ``run_auto_davey.py`` unless ``--allow-review-scoring`` is used.
    New locator-v2 metrics are read with ``getattr`` so legacy objects remain usable.
    """

    issues: list[ValidationIssue] = []
    seen_codes: set[str] = set()

    if image_bgr is None or image_bgr.size == 0:
        return InputValidationResult(
            status=InputStatus.INVALID,
            issues=(
                ValidationIssue(
                    code="UNREADABLE_IMAGE",
                    message="이미지를 읽을 수 없습니다.",
                    hard_reject=True,
                ),
            ),
            abdomen_area_ratio=0.0,
            brightness_mean=0.0,
            contrast_std=0.0,
        )

    if image_bgr.ndim not in (2, 3):
        return InputValidationResult(
            status=InputStatus.INVALID,
            issues=(
                ValidationIssue(
                    code="UNSUPPORTED_IMAGE_SHAPE",
                    message="지원하지 않는 이미지 배열 형식입니다.",
                    hard_reject=True,
                ),
            ),
            abdomen_area_ratio=0.0,
            brightness_mean=0.0,
            contrast_std=0.0,
        )

    height, width = image_bgr.shape[:2]
    brightness_mean, contrast_std = _image_statistics(image_bgr)

    if width < minimum_width or height < minimum_height:
        _add_issue(
            issues,
            seen_codes,
            code="IMAGE_TOO_SMALL",
            message=f"이미지 해상도가 너무 작습니다({width}x{height}).",
            hard_reject=False,
        )

    if brightness_mean < minimum_brightness:
        _add_issue(
            issues,
            seen_codes,
            code="IMAGE_TOO_DARK",
            message=f"이미지가 너무 어둡습니다(평균 밝기 {brightness_mean:.1f}).",
            hard_reject=False,
        )
    elif brightness_mean > maximum_brightness:
        _add_issue(
            issues,
            seen_codes,
            code="IMAGE_TOO_BRIGHT",
            message=f"이미지가 너무 밝습니다(평균 밝기 {brightness_mean:.1f}).",
            hard_reject=False,
        )

    if contrast_std < minimum_contrast_std:
        _add_issue(
            issues,
            seen_codes,
            code="IMAGE_LOW_CONTRAST",
            message=f"이미지 대비가 너무 낮습니다(표준편차 {contrast_std:.1f}).",
            hard_reject=False,
        )

    abdomen_mask = np.asarray(anatomy.abdomen_mask)
    if abdomen_mask.shape[:2] != (height, width):
        _add_issue(
            issues,
            seen_codes,
            code="ABDOMEN_MASK_SHAPE_MISMATCH",
            message="복부 마스크 크기가 입력 이미지와 다릅니다.",
            hard_reject=True,
        )
        abdomen_area_ratio = 0.0
    else:
        abdomen_area_ratio = float(np.count_nonzero(abdomen_mask)) / float(height * width)

    if not anatomy.valid:
        warning = anatomy.warning.strip() if anatomy.warning else "복부/배꼽 구조 검출이 유효하지 않습니다."
        _add_issue(
            issues,
            seen_codes,
            code="ANATOMY_INVALID",
            message=warning,
            hard_reject=True,
        )

    if abdomen_area_ratio < 0.08:
        _add_issue(
            issues,
            seen_codes,
            code="ABDOMEN_AREA_TOO_SMALL",
            message=f"복부 영역이 너무 작게 검출되었습니다({abdomen_area_ratio:.1%}).",
            hard_reject=True,
        )
    elif abdomen_area_ratio > 0.92:
        _add_issue(
            issues,
            seen_codes,
            code="ABDOMEN_AREA_TOO_LARGE",
            message=f"복부 영역이 사진 대부분을 차지합니다({abdomen_area_ratio:.1%}).",
            hard_reject=True,
        )

    if len(anatomy.abdomen_polygon) < 3:
        _add_issue(
            issues,
            seen_codes,
            code="ABDOMEN_POLYGON_INVALID",
            message="정상적인 복부 외곽선을 만들지 못했습니다.",
            hard_reject=True,
        )

    navel_x = int(anatomy.navel_x)
    navel_y = int(anatomy.navel_y)
    if not (0 <= navel_x < width and 0 <= navel_y < height):
        _add_issue(
            issues,
            seen_codes,
            code="NAVEL_OUT_OF_IMAGE",
            message="배꼽 좌표가 이미지 범위를 벗어났습니다.",
            hard_reject=True,
        )
    elif abdomen_mask.shape[:2] == (height, width) and abdomen_mask[navel_y, navel_x] == 0:
        _add_issue(
            issues,
            seen_codes,
            code="NAVEL_OUTSIDE_ABDOMEN",
            message="배꼽 좌표가 복부 영역 밖에 있습니다.",
            hard_reject=True,
        )

    if float(anatomy.abdomen_confidence) < minimum_abdomen_confidence:
        _add_issue(
            issues,
            seen_codes,
            code="LOW_ABDOMEN_CONFIDENCE",
            message=(
                "복부 검출 신뢰도가 낮습니다"
                f"({float(anatomy.abdomen_confidence):.3f} < {minimum_abdomen_confidence:.3f})."
            ),
            hard_reject=False,
        )

    if float(anatomy.navel_confidence) < minimum_navel_confidence:
        _add_issue(
            issues,
            seen_codes,
            code="LOW_NAVEL_CONFIDENCE",
            message=(
                "배꼽 검출 신뢰도가 낮습니다"
                f"({float(anatomy.navel_confidence):.3f} < {minimum_navel_confidence:.3f})."
            ),
            hard_reject=False,
        )

    margin_x = width * max(0.0, navel_edge_margin_ratio)
    margin_y = height * max(0.0, navel_edge_margin_ratio)
    if (
        0 <= navel_x < width
        and 0 <= navel_y < height
        and (
            navel_x < margin_x
            or navel_x > width - margin_x
            or navel_y < margin_y
            or navel_y > height - margin_y
        )
    ):
        _add_issue(
            issues,
            seen_codes,
            code="NAVEL_NEAR_IMAGE_EDGE",
            message="배꼽이 이미지 가장자리에 너무 가깝게 검출되었습니다.",
            hard_reject=False,
        )

    abdomen_solidity = _optional_float(anatomy, "abdomen_solidity")
    abdomen_extent = _optional_float(anatomy, "abdomen_extent")
    abdomen_core_width_ratio = _optional_float(anatomy, "abdomen_core_width_ratio")
    abdomen_component_count = _optional_int(anatomy, "abdomen_component_count")
    navel_boundary_margin = _optional_float(anatomy, "navel_boundary_margin_ratio")
    navel_concentration = _optional_float(anatomy, "navel_heatmap_concentration")
    navel_spread = _optional_float(anatomy, "navel_heatmap_spread_ratio")
    navel_candidate_area = _optional_float(anatomy, "navel_candidate_area_ratio")
    navel_candidate_components = _optional_int(anatomy, "navel_candidate_component_count")

    if abdomen_solidity is not None:
        if abdomen_solidity < minimum_abdomen_solidity_hard:
            _add_issue(
                issues,
                seen_codes,
                code="ABDOMEN_SHAPE_IMPLAUSIBLE",
                message=f"복부 mask 형태가 지나치게 불규칙합니다(solidity={abdomen_solidity:.3f}).",
                hard_reject=True,
            )
        elif abdomen_solidity < minimum_abdomen_solidity_review:
            _add_issue(
                issues,
                seen_codes,
                code="LOW_ABDOMEN_SOLIDITY",
                message=f"복부 mask 형태가 불안정합니다(solidity={abdomen_solidity:.3f}).",
                hard_reject=False,
            )

    if abdomen_extent is not None:
        if abdomen_extent < minimum_abdomen_extent_hard:
            _add_issue(
                issues,
                seen_codes,
                code="ABDOMEN_EXTENT_IMPLAUSIBLE",
                message=f"복부 mask가 bounding box 안에서 지나치게 성깁니다(extent={abdomen_extent:.3f}).",
                hard_reject=True,
            )
        elif abdomen_extent < minimum_abdomen_extent_review:
            _add_issue(
                issues,
                seen_codes,
                code="LOW_ABDOMEN_EXTENT",
                message=f"복부 mask 외곽이 불안정합니다(extent={abdomen_extent:.3f}).",
                hard_reject=False,
            )

    if abdomen_core_width_ratio is not None:
        if abdomen_core_width_ratio < minimum_abdomen_core_width_ratio_hard:
            _add_issue(
                issues,
                seen_codes,
                code="ABDOMEN_NARROW_BOTTLENECK",
                message=(
                    "복부 mask 중앙에 비정상적으로 좁은 연결부가 있습니다"
                    f"(core width={abdomen_core_width_ratio:.3f})."
                ),
                hard_reject=True,
            )
        elif abdomen_core_width_ratio < minimum_abdomen_core_width_ratio_review:
            _add_issue(
                issues,
                seen_codes,
                code="ABDOMEN_SHAPE_BOTTLENECK_REVIEW",
                message=(
                    "복부 mask 중앙 폭 변화가 큽니다"
                    f"(core width={abdomen_core_width_ratio:.3f})."
                ),
                hard_reject=False,
            )

    if abdomen_component_count is not None and abdomen_component_count >= 3:
        _add_issue(
            issues,
            seen_codes,
            code="MULTIPLE_ABDOMEN_COMPONENTS",
            message=f"복부 후보가 여러 덩이로 검출되었습니다(component={abdomen_component_count}).",
            hard_reject=False,
        )

    if navel_boundary_margin is not None:
        if navel_boundary_margin < minimum_navel_boundary_margin_ratio_hard:
            _add_issue(
                issues,
                seen_codes,
                code="NAVEL_TOO_CLOSE_TO_ABDOMEN_EDGE",
                message=(
                    "배꼽 후보가 복부 외곽선에 지나치게 가깝습니다"
                    f"(margin={navel_boundary_margin:.3f})."
                ),
                hard_reject=True,
            )
        elif navel_boundary_margin < minimum_navel_boundary_margin_ratio_review:
            _add_issue(
                issues,
                seen_codes,
                code="NAVEL_NEAR_ABDOMEN_EDGE",
                message=(
                    "배꼽 후보가 복부 외곽선과 너무 가깝습니다"
                    f"(margin={navel_boundary_margin:.3f})."
                ),
                hard_reject=False,
            )

    if navel_concentration is not None and navel_concentration < minimum_navel_heatmap_concentration:
        _add_issue(
            issues,
            seen_codes,
            code="NAVEL_HEATMAP_MULTIMODAL",
            message=(
                "배꼽 heatmap의 최고 후보가 충분히 집중되지 않았습니다"
                f"(concentration={navel_concentration:.3f})."
            ),
            hard_reject=False,
        )

    if navel_spread is not None and navel_spread > maximum_navel_heatmap_spread_ratio:
        _add_issue(
            issues,
            seen_codes,
            code="NAVEL_HEATMAP_TOO_DIFFUSE",
            message=f"배꼽 heatmap이 너무 넓게 퍼져 있습니다(spread={navel_spread:.3f}).",
            hard_reject=False,
        )

    if navel_candidate_area is not None and navel_candidate_area > maximum_navel_candidate_area_ratio:
        _add_issue(
            issues,
            seen_codes,
            code="NAVEL_CANDIDATE_TOO_LARGE",
            message=(
                "배꼽 후보 영역이 지나치게 큽니다"
                f"(area ratio={navel_candidate_area:.3f})."
            ),
            hard_reject=False,
        )

    if navel_candidate_components is not None and navel_candidate_components >= 3:
        _add_issue(
            issues,
            seen_codes,
            code="MULTIPLE_NAVEL_CANDIDATES",
            message=f"배꼽 후보 peak가 여러 곳에 존재합니다(component={navel_candidate_components}).",
            hard_reject=False,
        )

    if any(issue.hard_reject for issue in issues):
        status = InputStatus.INVALID
    elif issues:
        status = InputStatus.REVIEW
    else:
        status = InputStatus.VALID

    return InputValidationResult(
        status=status,
        issues=tuple(issues),
        abdomen_area_ratio=abdomen_area_ratio,
        brightness_mean=brightness_mean,
        contrast_std=contrast_std,
        abdomen_solidity=abdomen_solidity,
        abdomen_extent=abdomen_extent,
        abdomen_core_width_ratio=abdomen_core_width_ratio,
        abdomen_component_count=abdomen_component_count,
        navel_boundary_margin_ratio=navel_boundary_margin,
        navel_heatmap_concentration=navel_concentration,
        navel_heatmap_spread_ratio=navel_spread,
        navel_candidate_area_ratio=navel_candidate_area,
        navel_candidate_component_count=navel_candidate_components,
    )


def draw_validation_overlay(
    image_bgr: np.ndarray,
    anatomy: AnatomyLike,
    validation: InputValidationResult,
) -> np.ndarray:
    canvas = image_bgr.copy()

    if len(anatomy.abdomen_polygon) >= 3:
        polygon = np.asarray(anatomy.abdomen_polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)

    height, width = canvas.shape[:2]
    navel_x = int(anatomy.navel_x)
    navel_y = int(anatomy.navel_y)
    if 0 <= navel_x < width and 0 <= navel_y < height:
        cv2.circle(canvas, (navel_x, navel_y), 7, (255, 0, 255), thickness=2)

    status_text = f"INPUT: {validation.status.value.upper()}"
    status_color = (
        (0, 200, 0)
        if validation.status == InputStatus.VALID
        else ((0, 180, 255) if validation.status == InputStatus.REVIEW else (0, 0, 255))
    )
    cv2.putText(
        canvas,
        status_text,
        (14, min(34, max(18, height - 8))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        status_text,
        (14, min(34, max(18, height - 8))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        status_color,
        2,
        cv2.LINE_AA,
    )

    diagnostic = []
    if validation.abdomen_solidity is not None:
        diagnostic.append(f"sol={validation.abdomen_solidity:.2f}")
    if validation.abdomen_core_width_ratio is not None:
        diagnostic.append(f"core={validation.abdomen_core_width_ratio:.2f}")
    if validation.navel_boundary_margin_ratio is not None:
        diagnostic.append(f"navel-margin={validation.navel_boundary_margin_ratio:.2f}")
    if diagnostic:
        cv2.putText(
            canvas,
            "  ".join(diagnostic),
            (14, min(58, max(36, height - 8))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas
