import pytest
from airmouse.core.mouse import MouseController


def test_mouse_mapping_and_mirroring():
    # Screen 1000x1000, ROI: x in [0.2, 0.8], y in [0.2, 0.8]
    controller = MouseController(
        screen_size=(1000, 1000),
        roi_x_min=0.2,
        roi_x_max=0.8,
        roi_y_min=0.2,
        roi_y_max=0.8,
        dummy_mode=True,
    )

    # Center of ROI: norm_x = 0.5 (mirrored = 0.5), norm_y = 0.5
    sx, sy = controller.map_to_screen(0.5, 0.5)
    # 0.5 is exactly midway in [0.2, 0.8] -> screen center ~ 500
    assert abs(sx - 500) <= 2
    assert abs(sy - 500) <= 2

    # Left hand in camera: norm_x = 0.8
    # Mirrored x = 0.2 (left edge of ROI) -> screen_x should be 0
    sx, sy = controller.map_to_screen(0.8, 0.5)
    assert sx == 0

    # Right hand in camera: norm_x = 0.2
    # Mirrored x = 0.8 (right edge of ROI) -> screen_x should be 999
    sx, sy = controller.map_to_screen(0.2, 0.5)
    assert sx == 999

    # Clamping outside ROI
    sx_out, sy_out = controller.map_to_screen(0.0, 1.0)
    assert sx_out == 999
    assert sy_out == 999


def test_mouse_dummy_actions():
    controller = MouseController(screen_size=(800, 600), dummy_mode=True)
    sx, sy = controller.move_to(0.5, 0.5)
    assert controller.position == (sx, sy)

    # Actions in dummy mode should execute without exceptions
    controller.click("left")
    controller.click("right")
    controller.drag_start()
    assert controller._is_dragging
    controller.drag_stop()
    assert not controller._is_dragging
    controller.scroll(0.1, -0.2)
