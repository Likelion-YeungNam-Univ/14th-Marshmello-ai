from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskShapeMetrics:
    area_ratio: float
    solidity: float
    extent: float
    bbox_aspect_ratio: float
    core_width_ratio: float
    component_count: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int


@dataclass(frozen=True)
class NavelLocation:
    x: int
    y: int
    peak_confidence: float
    concentration: float
    spread_ratio: float
    candidate_area_ratio: float
    candidate_component_count: int


def create_gaussian_heatmap(
    height: int,
    width: int,
    x: float,
    y: float,
    sigma: float,
) -> np.ndarray:
    """Return a float32 Gaussian keypoint heatmap in [0, 1]."""
    if height <= 0 or width <= 0:
        raise ValueError("height와 width는 양수여야 합니다.")
    sigma = float(max(sigma, 1e-3))
    x = float(np.clip(x, 0.0, width - 1.0))
    y = float(np.clip(y, 0.0, height - 1.0))

    heatmap = np.zeros((height, width), dtype=np.float32)
    radius = max(2, int(np.ceil(4.0 * sigma)))
    x0 = max(0, int(np.floor(x)) - radius)
    x1 = min(width, int(np.ceil(x)) + radius + 1)
    y0 = max(0, int(np.floor(y)) - radius)
    y1 = min(height, int(np.ceil(y)) + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance_sq = (xx.astype(np.float32) - x) ** 2 + (yy.astype(np.float32) - y) ** 2
    patch = np.exp(-distance_sq / (2.0 * sigma * sigma)).astype(np.float32)
    maximum = float(patch.max())
    if maximum > 0:
        patch /= maximum
    heatmap[y0:y1, x0:x1] = patch
    return heatmap


def significant_component_count(mask: np.ndarray, minimum_area_ratio: float = 0.01) -> int:
    binary = (mask > 0).astype(np.uint8)
    number, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if number <= 1:
        return 0
    total_area = int(np.count_nonzero(binary))
    if total_area <= 0:
        return 0
    minimum_area = max(4, int(round(total_area * max(0.0, minimum_area_ratio))))
    count = 0
    for label in range(1, number):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            count += 1
    return count


def compute_mask_shape_metrics(mask: np.ndarray) -> MaskShapeMetrics:
    binary = (mask > 0).astype(np.uint8) * 255
    height, width = binary.shape[:2]
    image_area = float(max(1, height * width))
    nonzero = int(np.count_nonzero(binary))
    area_ratio = nonzero / image_area

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return MaskShapeMetrics(
            area_ratio=area_ratio,
            solidity=0.0,
            extent=0.0,
            bbox_aspect_ratio=0.0,
            core_width_ratio=0.0,
            component_count=0,
            bbox_x=0,
            bbox_y=0,
            bbox_width=0,
            bbox_height=0,
        )

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = contour_area / hull_area if hull_area > 0 else 0.0

    x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
    bbox_area = float(max(1, bbox_width * bbox_height))
    extent = contour_area / bbox_area
    bbox_aspect_ratio = bbox_width / float(max(1, bbox_height))

    # 중앙 60% 구간에서 행별 복부 폭을 측정합니다. 두 덩이가 좁은 목으로
    # 이어지는 figure-eight 형태나 서혜부 쪽으로 잘못 이어진 mask를 잡는 데 사용합니다.
    y_start = y + int(round(bbox_height * 0.20))
    y_end = y + int(round(bbox_height * 0.80))
    y_start = int(np.clip(y_start, 0, height))
    y_end = int(np.clip(y_end, y_start + 1, height))
    row_widths: list[int] = []
    for row in range(y_start, y_end):
        xs = np.flatnonzero(binary[row] > 0)
        if xs.size:
            row_widths.append(int(xs[-1] - xs[0] + 1))
    if row_widths and bbox_width > 0:
        core_width_ratio = float(np.percentile(row_widths, 10)) / float(bbox_width)
    else:
        core_width_ratio = 0.0

    return MaskShapeMetrics(
        area_ratio=area_ratio,
        solidity=float(solidity),
        extent=float(extent),
        bbox_aspect_ratio=float(bbox_aspect_ratio),
        core_width_ratio=float(core_width_ratio),
        component_count=significant_component_count(binary),
        bbox_x=int(x),
        bbox_y=int(y),
        bbox_width=int(bbox_width),
        bbox_height=int(bbox_height),
    )


def locate_navel_from_heatmap(
    probability: np.ndarray,
    abdomen_mask: np.ndarray,
    *,
    peak_ratio: float = 0.70,
    minimum_probability: float = 0.05,
) -> NavelLocation:
    """Locate a navel using the connected high-probability region around the peak.

    A single argmax pixel is noisy. This function first restricts the heatmap to the
    abdomen, takes the connected component containing the global peak among pixels
    above ``peak_ratio * peak``, and returns its probability-weighted centroid.
    """
    if probability.ndim != 2 or abdomen_mask.ndim != 2:
        raise ValueError("probability와 abdomen_mask는 2차원 배열이어야 합니다.")
    if probability.shape != abdomen_mask.shape:
        raise ValueError("probability와 abdomen_mask 크기가 같아야 합니다.")

    abdomen = abdomen_mask > 0
    abdomen_pixels = int(np.count_nonzero(abdomen))
    if abdomen_pixels == 0:
        return NavelLocation(0, 0, 0.0, 0.0, 1.0, 1.0, 0)

    constrained = np.asarray(probability, dtype=np.float32).copy()
    constrained[~abdomen] = 0.0
    flat_index = int(np.argmax(constrained))
    peak_y, peak_x = np.unravel_index(flat_index, constrained.shape)
    peak = float(constrained[peak_y, peak_x])
    if peak <= 0.0:
        return NavelLocation(int(peak_x), int(peak_y), 0.0, 0.0, 1.0, 1.0, 0)

    threshold = max(float(minimum_probability), peak * float(np.clip(peak_ratio, 0.0, 1.0)))
    candidate = ((constrained >= threshold) & abdomen).astype(np.uint8)
    number, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)

    component_count = max(0, number - 1)
    selected_label = 0
    selected_mass = -1.0
    for label in range(1, number):
        component = labels == label
        mass = float(constrained[component].sum())
        if mass > selected_mass:
            selected_label = label
            selected_mass = mass

    if selected_label <= 0:
        selected = np.zeros_like(candidate, dtype=bool)
        selected[peak_y, peak_x] = True
    else:
        selected = labels == selected_label

    ys, xs = np.where(selected)
    weights = constrained[ys, xs].astype(np.float64)
    weight_sum = float(weights.sum())
    if xs.size == 0 or weight_sum <= 0.0:
        navel_x, navel_y = int(peak_x), int(peak_y)
        spread_ratio = 1.0
    else:
        navel_x = int(round(float(np.sum(xs * weights) / weight_sum)))
        navel_y = int(round(float(np.sum(ys * weights) / weight_sum)))
        variance_x = float(np.sum(((xs - navel_x) ** 2) * weights) / weight_sum)
        variance_y = float(np.sum(((ys - navel_y) ** 2) * weights) / weight_sum)
        spread_pixels = float(np.sqrt(max(0.0, variance_x + variance_y)))
        shape = compute_mask_shape_metrics(abdomen_mask)
        scale = float(max(1, min(shape.bbox_width, shape.bbox_height)))
        spread_ratio = spread_pixels / scale

    candidate_mass = float(constrained[candidate > 0].sum())
    selected_mass = float(constrained[selected].sum())
    concentration = selected_mass / candidate_mass if candidate_mass > 0 else 0.0
    candidate_area_ratio = float(np.count_nonzero(selected)) / float(abdomen_pixels)
    selected_peak = float(constrained[selected].max()) if np.any(selected) else peak

    return NavelLocation(
        x=int(np.clip(navel_x, 0, probability.shape[1] - 1)),
        y=int(np.clip(navel_y, 0, probability.shape[0] - 1)),
        peak_confidence=selected_peak,
        concentration=float(concentration),
        spread_ratio=float(spread_ratio),
        candidate_area_ratio=float(candidate_area_ratio),
        candidate_component_count=int(component_count),
    )


def navel_boundary_margin_ratio(
    abdomen_mask: np.ndarray,
    navel_x: int,
    navel_y: int,
) -> float:
    binary = (abdomen_mask > 0).astype(np.uint8)
    height, width = binary.shape[:2]
    if not (0 <= navel_x < width and 0 <= navel_y < height):
        return 0.0
    if binary[navel_y, navel_x] == 0:
        return 0.0

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    shape = compute_mask_shape_metrics(binary)
    normalizer = float(max(1, min(shape.bbox_width, shape.bbox_height)))
    return float(distance[navel_y, navel_x]) / normalizer
