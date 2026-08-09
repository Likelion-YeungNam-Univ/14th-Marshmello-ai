from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from settings import CHECKPOINT_DIR, IMAGE_EXTENSIONS, NEW_IMAGES_DIR, OUTPUT_DIR
from unet_core import (
    create_model,
    letterbox_rgb,
    restore_probability_to_original,
    rgb_to_model_tensor,
)


QUADRANT_NAMES = {
    "left_upper": "좌상",
    "right_upper": "우상",
    "left_lower": "좌하",
    "right_lower": "우하",
}


@dataclass
class QuadrantResult:
    count: int
    score: int
    accepted_components: list[dict[str, int]]


@dataclass
class DaveyResult:
    image_name: str
    navel_x: int
    navel_y: int
    abdomen_roi_points: list[list[int]]
    threshold: float
    left_upper: QuadrantResult
    right_upper: QuadrantResult
    left_lower: QuadrantResult
    right_lower: QuadrantResult
    total_score: int


def collect_input_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(
            file
            for file in path.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def davey_quadrant_score(count: int) -> int:
    if count == 0:
        return 0
    if count <= 3:
        return 1
    return 2


def load_navel_points(csv_path: Path) -> dict[str, tuple[int, int]]:
    points: dict[str, tuple[int, int]] = {}
    if not csv_path.exists():
        return points

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                points[row["image_name"]] = (int(row["navel_x"]), int(row["navel_y"]))
            except (KeyError, TypeError, ValueError):
                continue
    return points


def save_navel_points(csv_path: Path, points: dict[str, tuple[int, int]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_name", "navel_x", "navel_y"])
        writer.writeheader()
        for image_name in sorted(points):
            x, y = points[image_name]
            writer.writerow({"image_name": image_name, "navel_x": x, "navel_y": y})


def load_abdomen_rois(json_path: Path) -> dict[str, list[tuple[int, int]]]:
    """이미지별 복부 ROI 다각형 좌표를 불러옵니다."""
    if not json_path.exists():
        return {}

    try:
        with json_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    rois: dict[str, list[tuple[int, int]]] = {}
    if not isinstance(raw, dict):
        return rois

    for image_name, points in raw.items():
        if not isinstance(image_name, str) or not isinstance(points, list):
            continue

        parsed: list[tuple[int, int]] = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                continue
            try:
                parsed.append((int(point[0]), int(point[1])))
            except (TypeError, ValueError):
                continue

        if len(parsed) >= 3:
            rois[image_name] = parsed

    return rois


def save_abdomen_rois(
    json_path: Path,
    rois: dict[str, list[tuple[int, int]]],
) -> None:
    """복부 ROI 좌표를 JSON으로 저장합니다."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        image_name: [[int(x), int(y)] for x, y in points]
        for image_name, points in sorted(rois.items())
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, ensure_ascii=False, indent=2)


def resize_for_display(image: np.ndarray, max_width: int = 1200, max_height: int = 800) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image.copy(), scale
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def select_navel_interactively(image_bgr: np.ndarray, image_name: str) -> tuple[int, int]:
    display_base, scale = resize_for_display(image_bgr)
    state: dict[str, tuple[int, int] | None] = {"point": None}
    window_name = f"Select navel - {image_name}"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            original_x = int(round(x / scale))
            original_y = int(round(y / scale))
            original_x = min(max(original_x, 0), image_bgr.shape[1] - 1)
            original_y = min(max(original_y, 0), image_bgr.shape[0] - 1)
            state["point"] = (original_x, original_y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\n배꼽 중앙을 마우스 왼쪽 버튼으로 클릭하세요.")
    print("Enter/Space: 확정, R: 다시 선택, Esc/Q: 취소")

    while True:
        canvas = display_base.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 58), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            "Click navel center | Enter: confirm | R: reset | Esc: cancel",
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        point = state["point"]
        if point is not None:
            display_x = int(round(point[0] * scale))
            display_y = int(round(point[1] * scale))
            cv2.circle(canvas, (display_x, display_y), 9, (0, 255, 255), 2)
            cv2.line(canvas, (display_x - 16, display_y), (display_x + 16, display_y), (0, 255, 255), 2)
            cv2.line(canvas, (display_x, display_y - 16), (display_x, display_y + 16), (0, 255, 255), 2)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32) and state["point"] is not None:
            selected = state["point"]
            cv2.destroyWindow(window_name)
            assert selected is not None
            return selected
        if key in (ord("r"), ord("R")):
            state["point"] = None
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("배꼽 선택을 취소했습니다.")


def select_abdomen_polygon_interactively(
    image_bgr: np.ndarray,
    image_name: str,
) -> list[tuple[int, int]]:
    """복부 피부 영역의 외곽을 다각형으로 선택합니다."""
    display_base, scale = resize_for_display(image_bgr)
    points: list[tuple[int, int]] = []
    window_name = f"Select abdomen ROI - {image_name}"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            original_x = int(round(x / scale))
            original_y = int(round(y / scale))
            original_x = min(max(original_x, 0), image_bgr.shape[1] - 1)
            original_y = min(max(original_y, 0), image_bgr.shape[0] - 1)
            points.append((original_x, original_y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\n복부 피부 영역의 바깥선을 따라 점을 찍으세요.")
    print("왼쪽 클릭: 점 추가, 오른쪽 클릭/U: 마지막 점 취소")
    print("Enter/Space: 확정, R: 전체 초기화, Esc/Q: 취소")
    print("옷, 배경, 팔, 침대 등은 다각형 밖에 두세요.")

    while True:
        canvas = display_base.copy()
        display_points = np.array(
            [
                [int(round(x * scale)), int(round(y * scale))]
                for x, y in points
            ],
            dtype=np.int32,
        )

        if len(display_points) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [display_points], (60, 160, 255))
            canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)
            cv2.polylines(canvas, [display_points], True, (0, 255, 255), 2)
        elif len(display_points) >= 2:
            cv2.polylines(canvas, [display_points], False, (0, 255, 255), 2)

        for point in display_points:
            cv2.circle(canvas, tuple(point), 5, (255, 0, 255), -1)

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            "L-click:add | R-click/U:undo | Enter:confirm | R:reset | Esc:cancel",
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):
            if len(points) < 3:
                print("점이 최소 3개 필요합니다.")
                continue
            selected = points.copy()
            cv2.destroyWindow(window_name)
            return selected
        if key in (ord("u"), ord("U"), 8) and points:
            points.pop()
        if key in (ord("r"), ord("R")):
            points.clear()
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("복부 ROI 선택을 취소했습니다.")


def polygon_to_mask(
    image_shape: tuple[int, int],
    points: list[tuple[int, int]],
) -> np.ndarray:
    """다각형 좌표를 0/255 복부 ROI 마스크로 변환합니다."""
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.asarray(points, dtype=np.int32)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [polygon], 255)
    return mask


def shrink_roi_mask(roi_mask: np.ndarray, erode_pixels: int) -> np.ndarray:
    """복부 경계 바로 바깥의 오검출을 줄이기 위해 ROI를 조금 안쪽으로 줄입니다."""
    if erode_pixels <= 0:
        return roi_mask

    kernel_size = erode_pixels * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.erode(roi_mask, kernel, iterations=1)


def predict_probability(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    letterboxed, metadata = letterbox_rgb(image_rgb, image_size)
    tensor = rgb_to_model_tensor(letterboxed).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probability_square = torch.sigmoid(logits)[0, 0].cpu().numpy()

    return restore_probability_to_original(probability_square, metadata)


def clean_binary_mask(mask: np.ndarray, connect_pixels: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if connect_pixels <= 1:
        return binary

    kernel_size = connect_pixels if connect_pixels % 2 == 1 else connect_pixels + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


def quadrant_bounds(width: int, height: int, navel_x: int, navel_y: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "left_upper": (0, 0, navel_x, navel_y),
        "right_upper": (navel_x, 0, width, navel_y),
        "left_lower": (0, navel_y, navel_x, height),
        "right_lower": (navel_x, navel_y, width, height),
    }


def count_components(
    quadrant_mask: np.ndarray,
    min_area: int,
    min_length: int,
) -> tuple[int, list[dict[str, int]]]:
    if quadrant_mask.size == 0:
        return 0, []

    binary = (quadrant_mask > 0).astype(np.uint8)
    number, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    accepted: list[dict[str, int]] = []
    for label_id in range(1, number):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        longest_side = max(width, height)

        if area < min_area:
            continue
        if longest_side < min_length:
            continue

        accepted.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
            }
        )

    return len(accepted), accepted


def analyze_quadrants(
    cleaned_mask: np.ndarray,
    navel_x: int,
    navel_y: int,
    min_area: int,
    min_length: int,
) -> dict[str, QuadrantResult]:
    height, width = cleaned_mask.shape[:2]
    results: dict[str, QuadrantResult] = {}

    for name, (x1, y1, x2, y2) in quadrant_bounds(width, height, navel_x, navel_y).items():
        crop = cleaned_mask[y1:y2, x1:x2]
        count, components = count_components(crop, min_area=min_area, min_length=min_length)

        global_components = []
        for component in components:
            global_component = component.copy()
            global_component["x"] += x1
            global_component["y"] += y1
            global_components.append(global_component)

        results[name] = QuadrantResult(
            count=count,
            score=davey_quadrant_score(count),
            accepted_components=global_components,
        )

    return results


def draw_result(
    image_bgr: np.ndarray,
    cleaned_mask: np.ndarray,
    abdomen_roi_mask: np.ndarray,
    navel_x: int,
    navel_y: int,
    quadrant_results: dict[str, QuadrantResult],
    total_score: int,
) -> np.ndarray:
    visualization = image_bgr.copy()

    red_layer = np.zeros_like(visualization)
    red_layer[:, :, 2] = 255
    selected = cleaned_mask > 0
    visualization[selected] = (
        0.55 * visualization[selected] + 0.45 * red_layer[selected]
    ).astype(np.uint8)

    roi_contours, _ = cv2.findContours(
        abdomen_roi_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(visualization, roi_contours, -1, (255, 255, 0), 2)

    height, width = visualization.shape[:2]
    cv2.line(visualization, (navel_x, 0), (navel_x, height - 1), (0, 255, 255), 2)
    cv2.line(visualization, (0, navel_y), (width - 1, navel_y), (0, 255, 255), 2)
    cv2.circle(visualization, (navel_x, navel_y), 8, (255, 0, 255), -1)

    for result in quadrant_results.values():
        for index, component in enumerate(result.accepted_components, start=1):
            x = component["x"]
            y = component["y"]
            w = component["width"]
            h = component["height"]
            cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(
                visualization,
                str(index),
                (x, max(16, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    positions = {
        "left_upper": (18, 84),
        "right_upper": (navel_x + 18, 84),
        "left_lower": (18, min(height - 18, navel_y + 46)),
        "right_lower": (navel_x + 18, min(height - 18, navel_y + 46)),
    }

    for name, result in quadrant_results.items():
        x, y = positions[name]
        text = f"{QUADRANT_NAMES[name]} count={result.count}, score={result.score}"
        cv2.putText(
            visualization,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            visualization,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

    title = f"Davey's score = {total_score} / 8"
    cv2.rectangle(visualization, (0, 0), (min(width - 1, 430), 52), (0, 0, 0), -1)
    cv2.putText(
        visualization,
        title,
        (14, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return visualization


def append_summary_csv(csv_path: Path, result: DaveyResult) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = [
        "image_name",
        "navel_x",
        "navel_y",
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

    row = {
        "image_name": result.image_name,
        "navel_x": result.navel_x,
        "navel_y": result.navel_y,
        "left_upper_count": result.left_upper.count,
        "left_upper_score": result.left_upper.score,
        "right_upper_count": result.right_upper.count,
        "right_upper_score": result.right_upper.score,
        "left_lower_count": result.left_lower.count,
        "left_lower_score": result.left_lower.score,
        "right_lower_count": result.right_lower.count,
        "right_lower_score": result.right_lower.score,
        "total_score": result.total_score,
    }

    with csv_path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="U-Net 임신선 마스크를 Davey's score로 변환합니다."
    )
    parser.add_argument("--input", type=Path, default=NEW_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "davey_results")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="이미 만들어진 마스크 폴더를 사용하려면 지정하세요.",
    )
    parser.add_argument("--navel-csv", type=Path, default=Path("navel_points.csv"))
    parser.add_argument(
        "--abdomen-roi-json",
        type=Path,
        default=Path("abdomen_rois.json"),
        help="이미지별 복부 ROI 다각형 좌표를 저장하는 JSON 파일.",
    )
    parser.add_argument(
        "--roi-mode",
        choices=("manual", "full"),
        default="manual",
        help="manual은 복부 외곽을 직접 지정하고, full은 전체 이미지를 사용합니다.",
    )
    parser.add_argument(
        "--reset-roi",
        action="store_true",
        help="저장된 ROI가 있어도 다시 선택합니다.",
    )
    parser.add_argument(
        "--roi-erode",
        type=int,
        default=3,
        help="복부 ROI 경계를 안쪽으로 줄일 픽셀 수. 경계 오검출이 많으면 키우세요.",
    )
    parser.add_argument("--navel-x", type=int, default=None)
    parser.add_argument("--navel-y", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--connect-pixels",
        type=int,
        default=5,
        help="끊어진 마스크 조각을 연결하는 커널 크기. 1이면 연결하지 않음.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=25,
        help="임신선 후보로 인정할 최소 픽셀 면적.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="임신선 후보로 인정할 bounding box 최소 긴 변 길이.",
    )
    args = parser.parse_args()

    input_images = collect_input_images(args.input)
    if not input_images:
        args.input.mkdir(parents=True, exist_ok=True)
        raise RuntimeError(f"입력 이미지가 없습니다: {args.input}")

    if (args.navel_x is None) != (args.navel_y is None):
        raise ValueError("--navel-x와 --navel-y는 반드시 함께 지정해야 합니다.")
    if args.navel_x is not None and len(input_images) != 1:
        raise ValueError("--navel-x/--navel-y는 입력 이미지가 한 장일 때만 사용할 수 있습니다.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: torch.nn.Module | None = None
    image_size = 512
    threshold = 0.5

    if args.mask_dir is None:
        checkpoint_path = CHECKPOINT_DIR / "best_unet.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "checkpoints/best_unet.pt가 없습니다. 먼저 python train.py를 실행하세요."
            )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        image_size = int(checkpoint.get("image_size", 512))
        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else float(checkpoint.get("threshold", 0.5))
        )
        encoder_name = str(checkpoint.get("encoder_name", "resnet34"))
        model = create_model(encoder_name, use_imagenet_weights=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    elif args.threshold is not None:
        threshold = float(args.threshold)

    args.output.mkdir(parents=True, exist_ok=True)
    mask_output_dir = args.output / "masks"
    cleaned_output_dir = args.output / "cleaned_masks"
    abdomen_mask_output_dir = args.output / "abdomen_masks"
    roi_applied_output_dir = args.output / "roi_applied_masks"
    visualization_output_dir = args.output / "visualizations"
    json_output_dir = args.output / "json"
    for directory in (
        mask_output_dir,
        abdomen_mask_output_dir,
        roi_applied_output_dir,
        cleaned_output_dir,
        visualization_output_dir,
        json_output_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    navel_points = load_navel_points(args.navel_csv)
    abdomen_rois = load_abdomen_rois(args.abdomen_roi_json)
    summary_csv = args.output / "davey_scores.csv"

    for image_path in input_images:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"읽기 실패: {image_path}")
            continue

        if args.mask_dir is not None:
            mask_path = args.mask_dir / f"{image_path.stem}.png"
            mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_u8 is None:
                print(f"마스크 없음, 건너뜀: {mask_path}")
                continue
            if mask_u8.shape != image_bgr.shape[:2]:
                mask_u8 = cv2.resize(
                    mask_u8,
                    (image_bgr.shape[1], image_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
        else:
            assert model is not None
            probability = predict_probability(image_bgr, model, device, image_size)
            mask_u8 = (probability >= threshold).astype(np.uint8) * 255

        cv2.imwrite(str(mask_output_dir / f"{image_path.stem}.png"), mask_u8)

        if args.roi_mode == "full":
            height, width = image_bgr.shape[:2]
            abdomen_points = [
                (0, 0),
                (width - 1, 0),
                (width - 1, height - 1),
                (0, height - 1),
            ]
        elif image_path.name in abdomen_rois and not args.reset_roi:
            abdomen_points = abdomen_rois[image_path.name]
        else:
            abdomen_points = select_abdomen_polygon_interactively(
                image_bgr,
                image_path.name,
            )
            abdomen_rois[image_path.name] = abdomen_points
            save_abdomen_rois(args.abdomen_roi_json, abdomen_rois)

        abdomen_roi_mask = polygon_to_mask(
            image_bgr.shape[:2],
            abdomen_points,
        )
        abdomen_roi_mask = shrink_roi_mask(
            abdomen_roi_mask,
            erode_pixels=max(0, args.roi_erode),
        )
        cv2.imwrite(
            str(abdomen_mask_output_dir / f"{image_path.stem}.png"),
            abdomen_roi_mask,
        )

        roi_applied_mask = cv2.bitwise_and(mask_u8, abdomen_roi_mask)
        cv2.imwrite(
            str(roi_applied_output_dir / f"{image_path.stem}.png"),
            roi_applied_mask,
        )

        cleaned_mask = clean_binary_mask(
            roi_applied_mask,
            connect_pixels=args.connect_pixels,
        )
        # closing 연산으로 ROI 바깥에 다시 생길 수 있는 픽셀을 한 번 더 제거합니다.
        cleaned_mask = cv2.bitwise_and(cleaned_mask, abdomen_roi_mask)
        cv2.imwrite(
            str(cleaned_output_dir / f"{image_path.stem}.png"),
            cleaned_mask,
        )

        if args.navel_x is not None and args.navel_y is not None:
            navel_x, navel_y = args.navel_x, args.navel_y
        elif image_path.name in navel_points:
            navel_x, navel_y = navel_points[image_path.name]
        else:
            navel_x, navel_y = select_navel_interactively(image_bgr, image_path.name)
            navel_points[image_path.name] = (navel_x, navel_y)
            save_navel_points(args.navel_csv, navel_points)

        navel_x = min(max(int(navel_x), 1), image_bgr.shape[1] - 1)
        navel_y = min(max(int(navel_y), 1), image_bgr.shape[0] - 1)

        if abdomen_roi_mask[navel_y, navel_x] == 0:
            print(
                f"경고: {image_path.name}의 배꼽 좌표가 복부 ROI 밖에 있습니다. "
                "ROI를 다시 지정하려면 --reset-roi 옵션을 사용하세요."
            )

        quadrant_results = analyze_quadrants(
            cleaned_mask,
            navel_x=navel_x,
            navel_y=navel_y,
            min_area=args.min_area,
            min_length=args.min_length,
        )
        total_score = sum(result.score for result in quadrant_results.values())

        result = DaveyResult(
            image_name=image_path.name,
            navel_x=navel_x,
            navel_y=navel_y,
            abdomen_roi_points=[[int(x), int(y)] for x, y in abdomen_points],
            threshold=threshold,
            left_upper=quadrant_results["left_upper"],
            right_upper=quadrant_results["right_upper"],
            left_lower=quadrant_results["left_lower"],
            right_lower=quadrant_results["right_lower"],
            total_score=total_score,
        )

        visualization = draw_result(
            image_bgr,
            cleaned_mask,
            abdomen_roi_mask,
            navel_x,
            navel_y,
            quadrant_results,
            total_score,
        )
        cv2.imwrite(
            str(visualization_output_dir / f"{image_path.stem}_davey.jpg"),
            visualization,
        )

        with (json_output_dir / f"{image_path.stem}.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(asdict(result), file, ensure_ascii=False, indent=2)

        append_summary_csv(summary_csv, result)

        print("\n" + "=" * 58)
        print(f"이미지: {image_path.name}")
        for name in ("left_upper", "right_upper", "left_lower", "right_lower"):
            quadrant = quadrant_results[name]
            print(
                f"{QUADRANT_NAMES[name]}: 임신선 후보 {quadrant.count}개 "
                f"-> {quadrant.score}점"
            )
        print(f"최종 Davey's score: {total_score}/8")
        print("=" * 58)

    print(f"\n완료: {args.output.resolve()}")
    print(f"배꼽 좌표: {args.navel_csv.resolve()}")
    print(f"복부 ROI 좌표: {args.abdomen_roi_json.resolve()}")


if __name__ == "__main__":
    main()
