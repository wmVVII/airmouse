import pytest
from airmouse.core.config import AppConfig
from airmouse.main import AirMouseApp


def test_end_to_end_pipeline():
    config = AppConfig()
    config.gui = False  # Headless mode for testing
    config.camera.target_fps = 30

    app = AirMouseApp(
        config=config,
        use_mock_camera=True,
        dummy_mouse=True,
    )
    app.initialize()

    # Run for 15 frames
    app.run(max_frames=15)

    assert not app.running
    app.stop()
