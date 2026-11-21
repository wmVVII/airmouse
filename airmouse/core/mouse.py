"""
High-performance cross-platform mouse controller using direct event dispatch.
Maps normalized camera ROI coordinates to screen resolution with zero artificial latency.
"""

from __future__ import annotations

import logging
import platform
import sys
from typing import Optional, Tuple

from pynput.mouse import Button, Controller

logger = logging.getLogger(__name__)


def get_screen_size() -> Tuple[int, int]:
    """
    Detects screen resolution cross-platform (Windows, macOS, Linux).
    Falls back gracefully to 1920x1080 if screen query fails.
    """
    os_name = platform.system()
    try:
        if os_name == "Windows":
            import ctypes
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        elif os_name == "Darwin":
            try:
                from AppKit import NSScreen
                frame = NSScreen.mainScreen().frame()
                return int(frame.size.width), int(frame.size.height)
            except ImportError:
                pass
        
        # Universal fallback using Tkinter
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception as e:
        logger.warning(f"Could not automatically detect screen size: {e}. Defaulting to 1920x1080.")

    return 1920, 1080


class MouseController:
    """
    Translates normalized camera coordinates into desktop cursor positions
    and performs mouse click, drag, and scroll operations.
    """

    def __init__(
        self,
        screen_size: Optional[Tuple[int, int]] = None,
        roi_x_min: float = 0.20,
        roi_x_max: float = 0.80,
        roi_y_min: float = 0.20,
        roi_y_max: float = 0.80,
        scroll_speed: float = 25.0,
        dummy_mode: bool = False,
    ) -> None:
        """
        Args:
            screen_size: (width, height) in pixels. Detected if None.
            roi_x_min, roi_x_max: Normalized camera X bounds for active interaction.
            roi_y_min, roi_y_max: Normalized camera Y bounds for active interaction.
            scroll_speed: Multiplier for scroll wheel velocity.
            dummy_mode: If True, simulates mouse without sending OS events (for tests).
        """
        self.screen_width, self.screen_height = screen_size or get_screen_size()
        self.roi_x_min = roi_x_min
        self.roi_x_max = roi_x_max
        self.roi_y_min = roi_y_min
        self.roi_y_max = roi_y_max
        self.scroll_speed = scroll_speed
        self.dummy_mode = dummy_mode

        self._mouse = Controller() if not dummy_mode else None
        self._is_dragging = False
        self._last_screen_pos: Tuple[int, int] = (self.screen_width // 2, self.screen_height // 2)

    def map_to_screen(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """
        Maps normalized coordinates to screen pixel coordinates:
        1. Mirrors X horizontally (natural selfie camera interaction).
        2. Remaps from active interaction ROI to full screen resolution.
        3. Clamps output to screen dimensions.
        """
        # Mirror horizontal coordinate so moving hand to right moves cursor to right
        mirrored_x = 1.0 - norm_x

        # Normalize relative to Active ROI
        roi_w = max(self.roi_x_max - self.roi_x_min, 1e-4)
        roi_h = max(self.roi_y_max - self.roi_y_min, 1e-4)

        u = (mirrored_x - self.roi_x_min) / roi_w
        v = (norm_y - self.roi_y_min) / roi_h

        # Clamp between 0.0 and 1.0
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))

        screen_x = int(u * (self.screen_width - 1))
        screen_y = int(v * (self.screen_height - 1))
        return screen_x, screen_y

    def move_to(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """Moves cursor to position mapped from normalized coordinates."""
        sx, sy = self.map_to_screen(norm_x, norm_y)
        self._last_screen_pos = (sx, sy)

        if not self.dummy_mode and self._mouse is not None:
            self._mouse.position = (sx, sy)

        return sx, sy

    def click(self, button: str = "left", count: int = 1) -> None:
        """Sends single or multiple clicks."""
        if self.dummy_mode or self._mouse is None:
            return

        b = Button.left if button == "left" else Button.right
        self._mouse.click(b, count)

    def drag_start(self) -> None:
        """Presses and holds the left mouse button."""
        if not self._is_dragging:
            self._is_dragging = True
            if not self.dummy_mode and self._mouse is not None:
                self._mouse.press(Button.left)

    def drag_stop(self) -> None:
        """Releases the left mouse button."""
        if self._is_dragging:
            self._is_dragging = False
            if not self.dummy_mode and self._mouse is not None:
                self._mouse.release(Button.left)

    def scroll(self, delta_x: float, delta_y: float) -> None:
        """
        Scrolls the mouse wheel.
        delta_y is inverted so moving hand up scrolls page up.
        """
        if self.dummy_mode or self._mouse is None:
            return

        # Invert vertical delta for natural scrolling feel
        scroll_steps_y = -int(delta_y * self.scroll_speed)
        scroll_steps_x = int(delta_x * self.scroll_speed)

        if scroll_steps_x != 0 or scroll_steps_y != 0:
            self._mouse.scroll(scroll_steps_x, scroll_steps_y)

    @property
    def position(self) -> Tuple[int, int]:
        """Returns current screen position."""
        if not self.dummy_mode and self._mouse is not None:
            return int(self._mouse.position[0]), int(self._mouse.position[1])
        return self._last_screen_pos
