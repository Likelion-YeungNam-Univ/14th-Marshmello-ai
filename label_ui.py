from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from davey_score import resize_for_display


HELP_PANEL_HEIGHT = 82
DISPLAY_MAX_WIDTH = 1200
DISPLAY_MAX_IMAGE_HEIGHT = 700


def _display_to_original(
    x: int,
    y: int,
    scale: float,
    image_shape: tuple[int, ...],
    display_image_height: int,
) -> tuple[int, int] | None:
    """Display 좌표를 원본 이미지 좌표로 변환합니다.

    사진 아래 help panel에서 발생한 클릭은 None을 반환해 무시합니다.
    """
    if y < 0 or y >= display_image_height:
        return None

    original_x = int(round(x / scale))
    original_y = int(round(y / scale))
    height, width = image_shape[:2]
    original_x = min(max(original_x, 0), width - 1)
    original_y = min(max(original_y, 0), height - 1)
    return original_x, original_y


def _with_help_panel(canvas: np.ndarray, lines: list[str]) -> np.ndarray:
    """이미지 픽셀을 덮지 않고 아래쪽 별도 패널에 안내 문구를 표시합니다."""
    panel_height = HELP_PANEL_HEIGHT if lines else 0
    if panel_height <= 0:
        return canvas

    output = cv2.copyMakeBorder(
        canvas,
        0,
        panel_height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(22, 22, 22),
    )

    y_positions = (27, 59)
    for index, text in enumerate(lines[:2]):
        cv2.putText(
            output,
            text,
            (12, y_positions[index]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return output


def select_navel_interactively(
    image_bgr: np.ndarray,
    image_name: str,
) -> tuple[int, int]:
    display_base, scale = resize_for_display(
        image_bgr,
        max_width=DISPLAY_MAX_WIDTH,
        max_height=DISPLAY_MAX_IMAGE_HEIGHT,
    )
    display_image_height = display_base.shape[0]
    state: dict[str, tuple[int, int] | None] = {"point": None}
    window_name = f"Select navel - {image_name}"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = _display_to_original(
            x,
            y,
            scale,
            image_bgr.shape,
            display_image_height,
        )
        if point is not None:
            state["point"] = point

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        display_base.shape[1],
        display_base.shape[0] + HELP_PANEL_HEIGHT,
    )
    cv2.setMouseCallback(window_name, on_mouse)

    print("\n배꼽 중앙을 마우스 왼쪽 버튼으로 클릭하세요.")
    print("안내문은 사진 아래에 표시되므로 이미지 영역을 가리지 않습니다.")
    print("Enter/Space: 확정, R: 다시 선택, Esc/Q: 취소")

    while True:
        canvas = display_base.copy()
        point = state["point"]
        if point is not None:
            display_x = int(round(point[0] * scale))
            display_y = int(round(point[1] * scale))
            cv2.circle(canvas, (display_x, display_y), 9, (0, 255, 255), 2)
            cv2.line(
                canvas,
                (display_x - 16, display_y),
                (display_x + 16, display_y),
                (0, 255, 255),
                2,
            )
            cv2.line(
                canvas,
                (display_x, display_y - 16),
                (display_x, display_y + 16),
                (0, 255, 255),
                2,
            )

        canvas = _with_help_panel(
            canvas,
            [
                "Click navel center | Enter/Space: confirm | R: reset",
                "Esc/Q: cancel | Help panel is outside the image",
            ],
        )
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
    """복부 피부 영역의 외곽을 다각형으로 선택합니다.

    조작 안내는 이미지 위에 그리지 않고 사진 아래 별도 패널에 표시합니다.
    """
    display_base, scale = resize_for_display(
        image_bgr,
        max_width=DISPLAY_MAX_WIDTH,
        max_height=DISPLAY_MAX_IMAGE_HEIGHT,
    )
    display_image_height = display_base.shape[0]
    points: list[tuple[int, int]] = []
    window_name = f"Select abdomen ROI - {image_name}"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        point = _display_to_original(
            x,
            y,
            scale,
            image_bgr.shape,
            display_image_height,
        )
        if point is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(point)
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        display_base.shape[1],
        display_base.shape[0] + HELP_PANEL_HEIGHT,
    )
    cv2.setMouseCallback(window_name, on_mouse)

    print("\n복부 피부 영역의 바깥선을 따라 점을 찍으세요.")
    print("안내문은 사진 아래에 표시되므로 복부 상단을 가리지 않습니다.")
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
            canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
            cv2.polylines(canvas, [display_points], True, (0, 255, 255), 2)
        elif len(display_points) >= 2:
            cv2.polylines(canvas, [display_points], False, (0, 255, 255), 2)

        for index, point in enumerate(display_points):
            color = (0, 255, 0) if index == 0 else (255, 0, 255)
            radius = 6 if index == 0 else 5
            cv2.circle(canvas, tuple(point), radius, color, -1)

        canvas = _with_help_panel(
            canvas,
            [
                "L-click:add | R-click/U:undo | Enter/Space:confirm",
                "R:reset all | Esc/Q:cancel | Green point = start",
            ],
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
