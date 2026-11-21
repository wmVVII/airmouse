"""
MediaPipe Hands detector optimized for real-time video tracking and low latency.
Supports modern MediaPipe Tasks (v1.x+) and legacy solutions API (v0.10.x).
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# Landmark indices for readability
class HandLandmark:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class Point3D:
    x: float
    y: float
    z: float


class HandLandmarks:
    """Wrapper around detected hand landmarks with helper methods."""

    def __init__(self, landmarks: List[Point3D], image_width: int, image_height: int) -> None:
        self.landmarks = landmarks
        self.width = image_width
        self.height = image_height

        # Calculate scale of the palm (wrist to middle MCP) for scale invariance
        p0 = self.landmarks[HandLandmark.WRIST]
        p9 = self.landmarks[HandLandmark.MIDDLE_FINGER_MCP]
        self.palm_scale = math.sqrt(
            (p0.x - p9.x) ** 2 + (p0.y - p9.y) ** 2 + (p0.z - p9.z) ** 2
        )
        if self.palm_scale < 1e-4:
            self.palm_scale = 1e-4

    def get_point(self, idx: int) -> Point3D:
        return self.landmarks[idx]

    def get_pixel(self, idx: int) -> Tuple[int, int]:
        pt = self.landmarks[idx]
        return int(pt.x * self.width), int(pt.y * self.height)

    def normalized_distance(self, idx1: int, idx2: int) -> float:
        """Computes Euclidean distance between two landmarks, normalized by palm scale."""
        p1 = self.landmarks[idx1]
        p2 = self.landmarks[idx2]
        dist = math.sqrt(
            (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2
        )
        return dist / self.palm_scale

    def is_extended(self, tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        """Determines if a finger is extended relative to the palm base."""
        wrist = self.landmarks[HandLandmark.WRIST]
        tip = self.landmarks[tip_idx]
        pip = self.landmarks[pip_idx]

        d_tip = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
        d_pip = (pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2
        return d_tip > d_pip


class HandDetector:
    """
    High-performance Hand Detector configured for minimum latency.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        max_num_hands: int = 1,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands
        self.model_path = model_path or self._resolve_model_path()
        self._landmarker = None
        self._legacy_hands = None
        self._start_time_ms: Optional[int] = None

        self._init_detector()

    def _resolve_model_path(self) -> str:
        """Finds or downloads the required hand_landmarker.task file."""
        candidates = [
            Path("models/hand_landmarker.task"),
            Path(__file__).resolve().parent.parent.parent / "models" / "hand_landmarker.task",
            Path.home() / ".cache" / "airmouse" / "hand_landmarker.task",
        ]
        for path in candidates:
            if path.exists() and path.stat().st_size > 1000000:
                return str(path.resolve())

        # Auto-download to first target
        target = candidates[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading HandLandmarker model to {target}...")
        try:
            urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, str(target))
            logger.info("HandLandmarker model downloaded successfully.")
            return str(target.resolve())
        except Exception as e:
            logger.warning(f"Failed to auto-download model from {MODEL_DOWNLOAD_URL}: {e}")
            return str(target.resolve())

    def _init_detector(self) -> None:
        import mediapipe as mp

        if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            from mediapipe.tasks.python import BaseOptions, vision

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"MediaPipe task model not found at {self.model_path}. "
                    "Please download hand_landmarker.task or check internet connectivity."
                )

            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            logger.info("Initialized MediaPipe Tasks HandLandmarker (Video Mode)")
        elif hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._legacy_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=0,
            )
            logger.info("Initialized legacy MediaPipe Solutions Hands")
        else:
            raise RuntimeError("Installed mediapipe package lacks both tasks.vision and solutions.hands.")

    def detect(self, bgr_frame: np.ndarray, timestamp_seconds: float) -> Optional[HandLandmarks]:
        """
        Processes a BGR image frame and returns hand landmarks if detected.
        
        Args:
            bgr_frame: Input frame from camera.
            timestamp_seconds: Capture timestamp in seconds.
        """
        h, w, _ = bgr_frame.shape
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        if self._landmarker is not None:
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(timestamp_seconds * 1000)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.hand_landmarks or len(result.hand_landmarks) == 0:
                return None

            raw_landmarks = result.hand_landmarks[0]
            points = [Point3D(lm.x, lm.y, lm.z) for lm in raw_landmarks]
            return HandLandmarks(points, w, h)

        elif self._legacy_hands is not None:
            results = self._legacy_hands.process(rgb_frame)
            if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) == 0:
                return None

            raw_landmarks = results.multi_hand_landmarks[0].landmark
            points = [Point3D(lm.x, lm.y, lm.z) for lm in raw_landmarks]
            return HandLandmarks(points, w, h)

        return None

    def close(self) -> None:
        """Release underlying detector resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._legacy_hands is not None:
            self._legacy_hands.close()
            self._legacy_hands = None
