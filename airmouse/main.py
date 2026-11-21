"""
Main entry point for Air Mouse application.
Orchestrates camera grabber, hand detector, 1€ filter, gesture engine, and mouse controller.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Optional

from airmouse.core.camera import MockCameraStream, ThreadedCameraStream
from airmouse.core.config import AppConfig
from airmouse.core.detector import HandDetector
from airmouse.core.filter import PointFilter2D
from airmouse.core.gestures import GestureEngine, GestureType
from airmouse.core.mouse import MouseController
from airmouse.ui.hud import HUDVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("airmouse")


class AirMouseApp:
    """Core Air Mouse runtime manager."""

    def __init__(self, config: AppConfig, use_mock_camera: bool = False, dummy_mouse: bool = False) -> None:
        self.config = config
        self.use_mock_camera = use_mock_camera
        self.dummy_mouse = dummy_mouse

        self.running = False
        self._camera = None
        self._detector: Optional[HandDetector] = None
        self._filter: Optional[PointFilter2D] = None
        self._gestures: Optional[GestureEngine] = None
        self._mouse: Optional[MouseController] = None
        self._hud: Optional[HUDVisualizer] = None

    def initialize(self) -> None:
        """Initializes all hardware and processing components."""
        logger.info("Initializing Air Mouse components...")

        # 1. Camera
        if self.use_mock_camera:
            logger.info("Starting in Mock Camera mode")
            self._camera = MockCameraStream(
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.target_fps,
            )
        else:
            self._camera = ThreadedCameraStream(
                camera_id=self.config.camera.camera_id,
                width=self.config.camera.width,
                height=self.config.camera.height,
                target_fps=self.config.camera.target_fps,
            )

        # 2. Hand detector
        self._detector = HandDetector(model_path=self.config.model_path)

        # 3. One Euro adaptive filter for 2D position
        self._filter = PointFilter2D(
            freq=self.config.filter.freq,
            mincutoff=self.config.filter.mincutoff,
            beta=self.config.filter.beta,
            dcutoff=self.config.filter.dcutoff,
        )

        # 4. Gesture recognition engine
        self._gestures = GestureEngine(
            pinch_press_thresh=self.config.gestures.pinch_press_thresh,
            pinch_release_thresh=self.config.gestures.pinch_release_thresh,
            drag_delay=self.config.gestures.drag_delay,
            double_click_delay=self.config.gestures.double_click_delay,
            scroll_threshold=self.config.gestures.scroll_threshold,
        )

        # 5. Cross-platform mouse controller
        self._mouse = MouseController(
            roi_x_min=self.config.roi.x_min,
            roi_x_max=self.config.roi.x_max,
            roi_y_min=self.config.roi.y_min,
            roi_y_max=self.config.roi.y_max,
            scroll_speed=self.config.gestures.scroll_speed,
            dummy_mode=self.dummy_mouse,
        )

        # 6. Visual HUD
        roi_bounds = (
            self.config.roi.x_min,
            self.config.roi.x_max,
            self.config.roi.y_min,
            self.config.roi.y_max,
        )
        self._hud = HUDVisualizer(roi=roi_bounds)
        self._hud.enabled = self.config.gui

        logger.info("Air Mouse initialized successfully.")

    def run(self, max_frames: Optional[int] = None) -> None:
        """Starts main processing loop."""
        if self._camera is None:
            self.initialize()

        self._camera.start()
        self.running = True

        logger.info("=" * 60)
        logger.info(" Air Mouse is now ACTIVE!")
        logger.info(" - Move Index Finger to aim cursor")
        logger.info(" - Pinch Index + Thumb to Click / Hold to Drag")
        logger.info(" - Pinch Middle + Thumb for Right Click")
        logger.info(" - Raise Index + Middle fingers together to Scroll")
        logger.info(" - Press [Space] to Pause/Clutch, [Q] to Quit")
        logger.info("=" * 60)

        frame_count = 0
        last_time = time.time()
        fps = 0.0

        try:
            while self.running:
                loop_start = time.time()
                grabbed, frame, timestamp = self._camera.read()

                if not grabbed or frame is None:
                    time.sleep(0.002)
                    continue

                frame_count += 1
                now = time.time()

                # Process hand tracking
                hand = self._detector.detect(frame, timestamp or now)
                event = self._gestures.process(hand, now)

                # Process cursor movement through One Euro Filter
                if event.type != GestureType.CLUTCH and hand is not None:
                    # Apply speed-adaptive filter to (x, y)
                    smooth_x, smooth_y = self._filter.filter(event.x, event.y, now)

                    if event.type in (GestureType.MOVE, GestureType.DRAGGING):
                        self._mouse.move_to(smooth_x, smooth_y)

                    if event.type == GestureType.LEFT_CLICK:
                        self._mouse.move_to(smooth_x, smooth_y)
                        self._mouse.click("left", count=1)
                        logger.debug("Left Click")

                    elif event.type == GestureType.DOUBLE_CLICK:
                        self._mouse.move_to(smooth_x, smooth_y)
                        self._mouse.click("left", count=2)
                        logger.debug("Double Click")

                    elif event.type == GestureType.RIGHT_CLICK:
                        self._mouse.move_to(smooth_x, smooth_y)
                        self._mouse.click("right", count=1)
                        logger.debug("Right Click")

                    elif event.type == GestureType.DRAG_START:
                        self._mouse.move_to(smooth_x, smooth_y)
                        self._mouse.drag_start()
                        logger.debug("Drag Start")

                    elif event.type == GestureType.DRAG_STOP:
                        self._mouse.drag_stop()
                        logger.debug("Drag Stop")

                    elif event.type == GestureType.SCROLL:
                        self._mouse.scroll(event.scroll_dx, event.scroll_dy)

                elif hand is None:
                    # Hand lost: reset filter memory so cursor does not jump when hand returns
                    self._filter.reset()

                # Calculate latency and FPS
                loop_duration = time.time() - loop_start
                latency_ms = loop_duration * 1000.0

                elapsed = now - last_time
                if elapsed >= 0.5:
                    fps = frame_count / elapsed
                    frame_count = 0
                    last_time = now

                # Render HUD
                if self.config.gui and self._hud is not None:
                    cont, key = self._hud.render(frame, hand, event, fps, latency_ms)
                    if not cont or key == ord("q"):
                        break
                    elif key == 32:  # Spacebar toggles clutch
                        clutched = self._gestures.toggle_clutch()
                        logger.info(f"Clutch mode: {'ON (Paused)' if clutched else 'OFF (Tracking)'}")
                    elif key == ord("h"):  # 'h' toggles HUD
                        self._hud.toggle()

                if max_frames is not None and frame_count >= max_frames:
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Shuts down camera, detector, and UI windows."""
        self.running = False
        if self._mouse is not None:
            self._mouse.drag_stop()
        if self._camera is not None:
            self._camera.stop()
        if self._detector is not None:
            self._detector.close()
        if self._hud is not None:
            self._hud.close()
        logger.info("Air Mouse stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-Performance Low-Latency Air Mouse")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=None, help="Camera capture width (e.g. 640)")
    parser.add_argument("--height", type=int, default=None, help="Camera capture height (e.g. 480)")
    parser.add_argument("--fps", type=int, default=None, help="Camera target FPS (e.g. 60)")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode without HUD window")
    parser.add_argument("--mock-camera", action="store_true", help="Use synthetic mock camera for testing")
    parser.add_argument("--dummy-mouse", action="store_true", help="Simulate mouse without moving real cursor")
    parser.add_argument("--sensitivity", type=float, default=None, help="Cursor sensitivity multiplier")
    parser.add_argument("--mincutoff", type=float, default=None, help="1€ filter mincutoff (lower = less tremor)")
    parser.add_argument("--beta", type=float, default=None, help="1€ filter beta (higher = less lag during motion)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.load(args.config)

    # CLI overrides
    if args.camera is not None:
        config.camera.camera_id = args.camera
    if args.width is not None:
        config.camera.width = args.width
    if args.height is not None:
        config.camera.height = args.height
    if args.fps is not None:
        config.camera.target_fps = args.fps
    if args.no_gui:
        config.gui = False
    if args.sensitivity is not None:
        config.sensitivity = args.sensitivity
    if args.mincutoff is not None:
        config.filter.mincutoff = args.mincutoff
    if args.beta is not None:
        config.filter.beta = args.beta

    app = AirMouseApp(
        config=config,
        use_mock_camera=args.mock_camera,
        dummy_mouse=args.dummy_mouse,
    )

    def sig_handler(sig, frame):
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    app.run()


if __name__ == "__main__":
    main()
