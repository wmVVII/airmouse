import pytest
from airmouse.core.detector import HandLandmark, HandLandmarks, Point3D
from airmouse.core.gestures import GestureEngine, GestureType


def make_mock_hand(
    wrist=(0.5, 0.8, 0.0),
    middle_mcp=(0.5, 0.5, 0.0),
    thumb_tip=(0.4, 0.3, 0.0),
    index_tip=(0.5, 0.2, 0.0),
    index_pip=(0.5, 0.4, 0.0),
    index_mcp=(0.5, 0.5, 0.0),
    middle_tip=(0.6, 0.6, 0.0),  # curled down by default
    middle_pip=(0.6, 0.4, 0.0),
    ring_tip=(0.7, 0.7, 0.0),    # curled down
    ring_pip=(0.7, 0.5, 0.0),
    ring_mcp=(0.7, 0.6, 0.0),
    pinky_tip=(0.8, 0.8, 0.0),   # curled down
    pinky_pip=(0.8, 0.6, 0.0),
    pinky_mcp=(0.8, 0.7, 0.0),
) -> HandLandmarks:
    """Creates synthetic HandLandmarks with customizable points."""
    pts = [Point3D(0.5, 0.5, 0.0) for _ in range(21)]
    pts[HandLandmark.WRIST] = Point3D(*wrist)
    pts[HandLandmark.MIDDLE_FINGER_MCP] = Point3D(*middle_mcp)
    pts[HandLandmark.THUMB_TIP] = Point3D(*thumb_tip)
    pts[HandLandmark.INDEX_FINGER_TIP] = Point3D(*index_tip)
    pts[HandLandmark.INDEX_FINGER_PIP] = Point3D(*index_pip)
    pts[HandLandmark.INDEX_FINGER_MCP] = Point3D(*index_mcp)
    pts[HandLandmark.MIDDLE_FINGER_TIP] = Point3D(*middle_tip)
    pts[HandLandmark.MIDDLE_FINGER_PIP] = Point3D(*middle_pip)
    pts[HandLandmark.RING_FINGER_TIP] = Point3D(*ring_tip)
    pts[HandLandmark.RING_FINGER_PIP] = Point3D(*ring_pip)
    pts[HandLandmark.RING_FINGER_MCP] = Point3D(*ring_mcp)
    pts[HandLandmark.PINKY_TIP] = Point3D(*pinky_tip)
    pts[HandLandmark.PINKY_PIP] = Point3D(*pinky_pip)
    pts[HandLandmark.PINKY_MCP] = Point3D(*pinky_mcp)
    return HandLandmarks(pts, 640, 480)


def test_gesture_move():
    engine = GestureEngine()
    # Hand with separated index and thumb
    hand = make_mock_hand(thumb_tip=(0.2, 0.2, 0.0), index_tip=(0.5, 0.2, 0.0))
    evt = engine.process(hand, timestamp=1.0)
    assert evt.type == GestureType.MOVE
    assert evt.x == 0.5
    assert evt.y == 0.2


def test_left_click_and_hysteresis():
    engine = GestureEngine(pinch_press_thresh=0.28, pinch_release_thresh=0.40)

    # 1. Open hand -> MOVE
    hand_open = make_mock_hand(thumb_tip=(0.3, 0.2, 0.0), index_tip=(0.6, 0.2, 0.0))
    evt = engine.process(hand_open, timestamp=1.0)
    assert evt.type == GestureType.MOVE

    # 2. Pinch thumb and index together (close distance) -> triggers pinch start
    hand_pinch = make_mock_hand(thumb_tip=(0.5, 0.2, 0.0), index_tip=(0.51, 0.2, 0.0))
    evt = engine.process(hand_pinch, timestamp=1.05)
    assert evt.is_pinching_index

    # 3. Release pinch quickly (within 0.1s) -> triggers LEFT_CLICK
    evt = engine.process(hand_open, timestamp=1.15)
    assert evt.type == GestureType.LEFT_CLICK


def test_drag_and_drop():
    engine = GestureEngine(drag_delay=0.25)
    hand_pinch = make_mock_hand(thumb_tip=(0.5, 0.2, 0.0), index_tip=(0.51, 0.2, 0.0))
    hand_open = make_mock_hand(thumb_tip=(0.3, 0.2, 0.0), index_tip=(0.6, 0.2, 0.0))

    # Pinch starts at t=1.0
    engine.process(hand_pinch, timestamp=1.0)

    # Still pinching at t=1.1 (under 0.25s)
    evt = engine.process(hand_pinch, timestamp=1.1)
    assert evt.is_pinching_index
    assert evt.type != GestureType.DRAGGING

    # Held pinch past drag_delay (t=1.3) -> DRAG_START
    evt = engine.process(hand_pinch, timestamp=1.3)
    assert evt.type == GestureType.DRAG_START

    # Continued hold -> DRAGGING
    evt = engine.process(hand_pinch, timestamp=1.35)
    assert evt.type == GestureType.DRAGGING

    # Released -> DRAG_STOP
    evt = engine.process(hand_open, timestamp=1.4)
    assert evt.type == GestureType.DRAG_STOP


def test_right_click():
    engine = GestureEngine(pinch_press_thresh=0.28, pinch_release_thresh=0.40)
    # Thumb and Middle close together
    hand_right_pinch = make_mock_hand(
        thumb_tip=(0.5, 0.2, 0.0),
        middle_tip=(0.51, 0.2, 0.0),
        index_tip=(0.8, 0.2, 0.0),
    )
    evt = engine.process(hand_right_pinch, timestamp=1.0)
    assert evt.type == GestureType.RIGHT_CLICK
    assert evt.is_pinching_middle


def test_scroll_mode():
    engine = GestureEngine()
    # Index and middle both extended, ring & pinky curled
    hand_scroll = make_mock_hand(
        thumb_tip=(0.2, 0.4, 0.0),
        index_tip=(0.48, 0.2, 0.0),
        index_pip=(0.48, 0.4, 0.0),
        middle_tip=(0.52, 0.2, 0.0),
        middle_pip=(0.52, 0.4, 0.0),
    )
    evt1 = engine.process(hand_scroll, timestamp=1.0)
    assert evt1.type == GestureType.SCROLL
    assert evt1.is_scrolling

    # Move fingers up for scrolling
    hand_scroll_moved = make_mock_hand(
        thumb_tip=(0.2, 0.4, 0.0),
        index_tip=(0.48, 0.15, 0.0),
        index_pip=(0.48, 0.4, 0.0),
        middle_tip=(0.52, 0.15, 0.0),
        middle_pip=(0.52, 0.4, 0.0),
    )
    evt2 = engine.process(hand_scroll_moved, timestamp=1.03)
    assert evt2.type == GestureType.SCROLL
    assert evt2.scroll_dy < 0.0  # moved upwards


def test_clutch_toggle():
    engine = GestureEngine()
    hand = make_mock_hand()
    assert not engine.is_clutched

    engine.toggle_clutch()
    assert engine.is_clutched
    evt = engine.process(hand, timestamp=1.0)
    assert evt.type == GestureType.CLUTCH
    assert evt.is_clutched

    engine.toggle_clutch()
    assert not engine.is_clutched
