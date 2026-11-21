"""
High-performance threaded camera stream for minimal input latency.

Drops buffered stale frames from OpenCV's video queue so the processing
pipeline always operates on the most recently captured physical frame.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ThreadedCameraStream:
    """
    Captures webcam frames continuously in a background thread to eliminate
    buffer latency and frame accumulation.
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 60,
    ) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_timestamp: float = 0.0
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._last_fps_calc_time: float = 0.0

    def start(self) -> "ThreadedCameraStream":
        """Opens camera and starts the background worker thread."""
        logger.info(f"Opening camera {self.camera_id} at {self.width}x{self.height} @ {self.target_fps}fps")
        self.cap = cv2.VideoCapture(self.camera_id)

        # Set properties (hardware-supported properties vary across OS/devices)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        # Minimize backend buffer size if supported (e.g. V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera {self.camera_id}. Please check device connection/index.")

        # Read actual negotiated properties
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps}fps")

        self._stopped.clear()
        self._thread = threading.Thread(target=self._update_loop, daemon=True, name="CameraGrabber")
        self._thread.start()
        return self

    def _update_loop(self) -> None:
        """Continuously grab the latest frame, immediately replacing the buffer."""
        self._last_fps_calc_time = time.time()
        fps_frame_count = 0

        while not self._stopped.is_set():
            if self.cap is None or not self.cap.isOpened():
                break

            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.005)
                continue

            now = time.time()
            with self._lock:
                self._latest_frame = frame
                self._frame_timestamp = now
                self._frame_count += 1

            fps_frame_count += 1
            elapsed = now - self._last_fps_calc_time
            if elapsed >= 1.0:
                self._fps = fps_frame_count / elapsed
                fps_frame_count = 0
                self._last_fps_calc_time = now

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Retrieves the latest available frame without blocking on camera hardware.
        
        Returns:
            (success, frame, timestamp)
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0.0
            return True, self._latest_frame.copy(), self._frame_timestamp

    @property
    def fps(self) -> float:
        """Measured hardware capture FPS."""
        return self._fps

    def stop(self) -> None:
        """Signals background thread to exit and releases camera."""
        self._stopped.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera stream stopped.")


class MockCameraStream:
    """Mock camera stream generating synthetic frames for testing and headless environments."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.target_fps = fps
        self._stopped = False
        self._frame_count = 0
        self._fps = float(fps)

    def start(self) -> "MockCameraStream":
        self._stopped = False
        return self

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if self._stopped:
            return False, None, 0.0
        now = time.time()
        self._frame_count += 1
        # Create a blank black/gradient frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Draw a moving mock test circle
        cx = int((self.width / 2) + 100 * np.sin(self._frame_count * 0.1))
        cy = int((self.height / 2) + 50 * np.cos(self._frame_count * 0.1))
        cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
        return True, frame, now

    @property
    def fps(self) -> float:
        return self._fps

    def stop(self) -> None:
        self._stopped = True
