"""
Scale-invariant gesture recognition state machine for mouse interactions.
Implements hysteresis debouncing, pinch detection, drag & drop, and scrolling.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from airmouse.core.detector import HandLandmark, HandLandmarks


class GestureType(enum.Enum):
    NONE = "none"
    MOVE = "move"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    DRAG_START = "drag_start"
    DRAGGING = "dragging"
    DRAG_STOP = "drag_stop"
    SCROLL = "scroll"
    CLUTCH = "clutch"


@dataclass
class GestureEvent:
    type: GestureType
    x: float
    y: float
    scroll_dx: float = 0.0
    scroll_dy: float = 0.0
    is_pinching_index: bool = False
    is_pinching_middle: bool = False
    is_scrolling: bool = False
    is_clutched: bool = False


class GestureEngine:
    """
    Evaluates hand landmarks over time to recognize cursor motion,
    clicks, drag & drop, and scrolling with hysteresis and debounce.
    """

    def __init__(
        self,
        pinch_press_thresh: float = 0.28,
        pinch_release_thresh: float = 0.40,
        drag_delay: float = 0.25,
        double_click_delay: float = 0.35,
        scroll_threshold: float = 0.35,
    ) -> None:
        self.pinch_press_thresh = pinch_press_thresh
        self.pinch_release_thresh = pinch_release_thresh
        self.drag_delay = drag_delay
        self.double_click_delay = double_click_delay
        self.scroll_threshold = scroll_threshold

        # State tracking for Left Pinch (Thumb + Index)
        self._index_pinched = False
        self._index_pinch_start_time = 0.0
        self._is_dragging = False
        self._last_left_click_time = 0.0
        self._locked_click_pos: Optional[Tuple[float, float]] = None

        # State tracking for Right Pinch (Thumb + Middle)
        self._middle_pinched = False

        # State tracking for Scrolling
        self._is_scrolling = False
        self._prev_scroll_y: Optional[float] = None
        self._prev_scroll_x: Optional[float] = None

        # Clutch state
        self._clutch_active = False

    def toggle_clutch(self) -> bool:
        """Toggles clutch/pause mode on or off."""
        self._clutch_active = not self._clutch_active
        return self._clutch_active

    @property
    def is_clutched(self) -> bool:
        return self._clutch_active

    def process(self, hand: Optional[HandLandmarks], timestamp: Optional[float] = None) -> GestureEvent:
        """
        Analyzes current hand pose and returns the active gesture event.
        """
        now = timestamp if timestamp is not None else time.time()

        if hand is None:
            # Hand lost: release any active drags
            if self._is_dragging:
                self._is_dragging = False
                self._index_pinched = False
                return GestureEvent(GestureType.DRAG_STOP, 0.5, 0.5)
            self._index_pinched = False
            self._middle_pinched = False
            self._is_scrolling = False
            self._prev_scroll_y = None
            return GestureEvent(GestureType.NONE, 0.5, 0.5)

        if self._clutch_active:
            return GestureEvent(GestureType.CLUTCH, 0.5, 0.5, is_clutched=True)

        # Retrieve key landmarks
        index_tip = hand.get_point(HandLandmark.INDEX_FINGER_TIP)
        cursor_x, cursor_y = index_tip.x, index_tip.y

        # Distances normalized by palm scale
        d_thumb_index = hand.normalized_distance(
            HandLandmark.THUMB_TIP, HandLandmark.INDEX_FINGER_TIP
        )
        d_thumb_middle = hand.normalized_distance(
            HandLandmark.THUMB_TIP, HandLandmark.MIDDLE_FINGER_TIP
        )
        d_index_middle = hand.normalized_distance(
            HandLandmark.INDEX_FINGER_TIP, HandLandmark.MIDDLE_FINGER_TIP
        )

        index_extended = hand.is_extended(
            HandLandmark.INDEX_FINGER_TIP,
            HandLandmark.INDEX_FINGER_PIP,
            HandLandmark.INDEX_FINGER_MCP,
        )
        middle_extended = hand.is_extended(
            HandLandmark.MIDDLE_FINGER_TIP,
            HandLandmark.MIDDLE_FINGER_PIP,
            HandLandmark.MIDDLE_FINGER_MCP,
        )
        ring_extended = hand.is_extended(
            HandLandmark.RING_FINGER_TIP,
            HandLandmark.RING_FINGER_PIP,
            HandLandmark.RING_FINGER_MCP,
        )
        pinky_extended = hand.is_extended(
            HandLandmark.PINKY_TIP,
            HandLandmark.PINKY_PIP,
            HandLandmark.PINKY_MCP,
        )

        # ---------------------------------------------------------------------
        # 1. LEFT PINCH (Click, Double Click, Drag & Drop) - Priority 1
        # ---------------------------------------------------------------------
        # Check pinch press transition
        if not self._index_pinched and d_thumb_index < self.pinch_press_thresh:
            self._index_pinched = True
            self._index_pinch_start_time = now
            # Stabilize cursor position at contact to prevent click-slip
            self._locked_click_pos = (cursor_x, cursor_y)

        # Check pinch release transition
        elif self._index_pinched and d_thumb_index > self.pinch_release_thresh:
            self._index_pinched = False
            click_pos = self._locked_click_pos or (cursor_x, cursor_y)
            self._locked_click_pos = None

            if self._is_dragging:
                self._is_dragging = False
                return GestureEvent(
                    GestureType.DRAG_STOP,
                    cursor_x,
                    cursor_y,
                    is_pinching_index=False,
                )
            else:
                # Single vs Double click
                if (now - self._last_left_click_time) < self.double_click_delay:
                    self._last_left_click_time = 0.0
                    return GestureEvent(
                        GestureType.DOUBLE_CLICK,
                        click_pos[0],
                        click_pos[1],
                        is_pinching_index=False,
                    )
                else:
                    self._last_left_click_time = now
                    return GestureEvent(
                        GestureType.LEFT_CLICK,
                        click_pos[0],
                        click_pos[1],
                        is_pinching_index=False,
                    )

        # While index pinch is active
        if self._index_pinched:
            hold_time = now - self._index_pinch_start_time
            if hold_time >= self.drag_delay:
                if not self._is_dragging:
                    self._is_dragging = True
                    return GestureEvent(
                        GestureType.DRAG_START,
                        cursor_x,
                        cursor_y,
                        is_pinching_index=True,
                    )
                else:
                    return GestureEvent(
                        GestureType.DRAGGING,
                        cursor_x,
                        cursor_y,
                        is_pinching_index=True,
                    )
            else:
                # Initial pinch window before drag triggers
                if self._locked_click_pos is not None:
                    cursor_x, cursor_y = self._locked_click_pos
                return GestureEvent(
                    GestureType.MOVE,
                    cursor_x,
                    cursor_y,
                    is_pinching_index=True,
                )

        # ---------------------------------------------------------------------
        # 2. RIGHT CLICK: Thumb + Middle Finger Pinch - Priority 2
        # ---------------------------------------------------------------------
        if not self._middle_pinched and d_thumb_middle < self.pinch_press_thresh:
            self._middle_pinched = True
            return GestureEvent(
                GestureType.RIGHT_CLICK,
                cursor_x,
                cursor_y,
                is_pinching_middle=True,
            )
        elif self._middle_pinched and d_thumb_middle > self.pinch_release_thresh:
            self._middle_pinched = False

        if self._middle_pinched:
            return GestureEvent(
                GestureType.MOVE,
                cursor_x,
                cursor_y,
                is_pinching_middle=True,
            )

        # ---------------------------------------------------------------------
        # 3. SCROLL MODE: Index & Middle extended, Ring & Pinky curled
        # ---------------------------------------------------------------------
        if (
            index_extended
            and middle_extended
            and not ring_extended
            and not pinky_extended
            and d_index_middle < self.scroll_threshold
        ):
            self._is_scrolling = True
            mid_y = (index_tip.y + hand.get_point(HandLandmark.MIDDLE_FINGER_TIP).y) / 2.0
            mid_x = (index_tip.x + hand.get_point(HandLandmark.MIDDLE_FINGER_TIP).x) / 2.0

            scroll_dy = 0.0
            scroll_dx = 0.0
            if self._prev_scroll_y is not None and self._prev_scroll_x is not None:
                scroll_dy = mid_y - self._prev_scroll_y
                scroll_dx = mid_x - self._prev_scroll_x

            self._prev_scroll_y = mid_y
            self._prev_scroll_x = mid_x

            return GestureEvent(
                GestureType.SCROLL,
                cursor_x,
                cursor_y,
                scroll_dx=scroll_dx,
                scroll_dy=scroll_dy,
                is_scrolling=True,
            )
        else:
            self._is_scrolling = False
            self._prev_scroll_y = None
            self._prev_scroll_x = None

        # ---------------------------------------------------------------------
        # 4. DEFAULT: Move Cursor
        # ---------------------------------------------------------------------
        return GestureEvent(
            GestureType.MOVE,
            cursor_x,
            cursor_y,
            is_pinching_index=False,
            is_pinching_middle=False,
        )
