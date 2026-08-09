from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import cv2
import numpy as np


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

    @property
    def can_score(self) -> bool:
        return self.status == InputStatus.VALID

    @property
    def warning(self) -> str:
        return "; ".join(issue.message for issue in self.issues)

    @property
    def reason_codes(self) -> list[str]:
        return [issue.code for issue in self.issues]


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
) -> InputValidationResult:
    """Validate whether an image is suitable for Davey's score inference.

    Hard failures become ``invalid_input``. Soft quality/confidence failures become
    ``review``. The caller should normally calculate Davey's score only for
    ``valid`` images.

    The thresholds are intentionally exposed as arguments because confidence
    values must ultimately be calibrated against the project's validation set.
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
        and (navel_x < margin_x or navel_x > width - margin_x or navel_y < margin_y or navel_y > height - margin_y)
    ):
        _add_issue(
            issues,
            seen_codes,
            code="NAVEL_NEAR_IMAGE_EDGE",
            message="배꼽이 이미지 가장자리에 너무 가깝게 검출되었습니다.",
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
    )


def draw_validation_overlay(
    image_bgr: np.ndarray,
    anatomy: AnatomyLike,
    validation: InputValidationResult,
) -> np.ndarray:
    """Create a lightweight diagnostic image without running stretch segmentation."""

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
        (0, 200, 0) if validation.status == InputStatus.VALID else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas
