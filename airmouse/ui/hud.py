"""
Lightweight visual overlay (HUD) for calibration, gesture feedback, and FPS metrics.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from airmouse.core.detector import HandLandmark, HandLandmarks
from airmouse.core.gestures import GestureEvent, GestureType

logger = logging.getLogger(__name__)


class HUDVisualizer:
    """
    Renders an interactive Heads-Up Display showing ROI bounds,
    hand skeleton, gesture states, and performance metrics.
    """

    def __init__(
        self,
        window_name: str = "Air Mouse - Live HUD",
        roi: Tuple[float, float, float, float] = (0.2, 0.8, 0.2, 0.8),
    ) -> None:
        self.window_name = window_name
        self.roi_x_min, self.roi_x_max, self.roi_y_min, self.roi_y_max = roi
        self.enabled = True
        self._window_created = False
        self._imshow_supported: Optional[bool] = None

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        if not self.enabled and self._window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self._window_created = False
        return self.enabled

    def render(
        self,
        frame: np.ndarray,
        hand: Optional[HandLandmarks],
        event: GestureEvent,
        fps: float,
        latency_ms: float,
    ) -> Tuple[bool, int]:
        """
        Draws visual feedback on the frame and presents it.
        Returns: (continue_running, key_pressed)
        """
        if not self.enabled:
            return True, -1

        h, w, _ = frame.shape
        # Mirror frame horizontally so the display acts like a mirror
        display_frame = cv2.flip(frame, 1)

        # 1. Draw Active ROI Box (mirrored)
        # In mirrored frame, camera x: norm_x -> (1 - norm_x)
        # So roi [x_min, x_max] becomes [1 - x_max, 1 - x_min]
        rx1 = int((1.0 - self.roi_x_max) * w)
        rx2 = int((1.0 - self.roi_x_min) * w)
        ry1 = int(self.roi_y_min * h)
        ry2 = int(self.roi_y_max * h)

        roi_color = (0, 255, 128) if not event.is_clutched else (0, 0, 255)
        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), roi_color, 2)
        cv2.putText(
            display_frame,
            "ACTIVE TRACKING ZONE",
            (rx1 + 10, ry1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            roi_color,
            1,
            cv2.LINE_AA,
        )

        # 2. Draw Hand Landmarks if detected
        if hand is not None:
            # Mirror x coordinates for display
            def to_disp_coords(idx: int) -> Tuple[int, int]:
                pt = hand.get_point(idx)
                return int((1.0 - pt.x) * w), int(pt.y * h)

            # Draw bones connections
            connections = [
                (HandLandmark.WRIST, HandLandmark.THUMB_CMC),
                (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
                (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
                (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
                (HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP),
                (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP),
                (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP),
                (HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP),
                (HandLandmark.WRIST, HandLandmark.MIDDLE_FINGER_MCP),
                (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP),
                (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP),
                (HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP),
                (HandLandmark.WRIST, HandLandmark.RING_FINGER_MCP),
                (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP),
                (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP),
                (HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
                (HandLandmark.WRIST, HandLandmark.PINKY_MCP),
                (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
                (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
                (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
            ]
            for start, end in connections:
                cv2.line(display_frame, to_disp_coords(start), to_disp_coords(end), (180, 180, 180), 1)

            # Draw key fingertips
            idx_pos = to_disp_coords(HandLandmark.INDEX_FINGER_TIP)
            thumb_pos = to_disp_coords(HandLandmark.THUMB_TIP)
            mid_pos = to_disp_coords(HandLandmark.MIDDLE_FINGER_TIP)

            # Color code index fingertip based on state
            if event.is_clutched:
                idx_color = (0, 0, 255)
            elif event.type in (GestureType.LEFT_CLICK, GestureType.DOUBLE_CLICK):
                idx_color = (0, 255, 255)  # Yellow
            elif event.type in (GestureType.DRAG_START, GestureType.DRAGGING):
                idx_color = (0, 165, 255)  # Orange
            elif event.type == GestureType.SCROLL:
                idx_color = (255, 0, 255)  # Purple
            elif event.type == GestureType.RIGHT_CLICK:
                idx_color = (255, 255, 0)  # Cyan
            else:
                idx_color = (0, 255, 0)    # Green

            # Draw fingertips
            cv2.circle(display_frame, thumb_pos, 7, (255, 100, 100), -1)
            cv2.circle(display_frame, mid_pos, 7, (100, 100, 255), -1)
            cv2.circle(display_frame, idx_pos, 10, idx_color, -1)
            cv2.circle(display_frame, idx_pos, 12, (255, 255, 255), 2)

        # 3. Status Bar & Metrics Overlay
        state_text = event.type.name
        if event.is_clutched:
            state_text = "PAUSED (CLUTCH)"
        
        cv2.rectangle(display_frame, (0, 0), (w, 40), (20, 20, 20), -1)
        status_line = (
            f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | State: {state_text}"
        )
        cv2.putText(
            display_frame,
            status_line,
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 4. Bottom controls guide
        cv2.rectangle(display_frame, (0, h - 30), (w, h), (20, 20, 20), -1)
        guide_line = "Controls: Pinch=Click | Hold Pinch=Drag | Middle Pinch=R-Click | 2-Fingers=Scroll | Space=Clutch | Q=Quit"
        cv2.putText(
            display_frame,
            guide_line,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Show window safely
        if self._imshow_supported is False:
            return True, -1

        try:
            cv2.imshow(self.window_name, display_frame)
            self._window_created = True
            self._imshow_supported = True
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False, key
            return True, key
        except Exception as e:
            if self._imshow_supported is None:
                logger.info(f"Display GUI not available in this environment ({e}). Running in headless mode.")
                self._imshow_supported = False
            return True, -1

    def close(self) -> None:
        if self._window_created:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self._window_created = False
