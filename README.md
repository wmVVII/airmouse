# 🖱️ FastRespown Air Mouse

High-performance, ultra-low latency, cross-platform **Air Mouse** that allows you to control your PC (Windows, macOS, Linux) with natural hand and finger gestures using a standard webcam.

Designed specifically for **instantaneous response ("fast respown")** and **jitter-free precision**, eliminating the lag and sluggishness typical of basic computer vision mouse implementations.

---

## ⚡ Why is it so fast?

Standard computer vision tracking apps feel sluggish because of frame buffering and naive smoothing. This project implements 4 core engineering optimizations:

1. **Threaded Buffer-Draining Camera Grabber**:
   Standard `cv2.VideoCapture` queues up frames in the OS video pipeline, introducing 100–300 ms of latency behind your real hand. Our dedicated background thread continuously drains stale frames and serves only the single freshest frame (<15 ms latency).
2. **One Euro (1€) Dynamic Filter**:
   Adapts its cutoff frequency based on movement speed:
   - When hand is still: low cutoff frequency suppresses micro-tremors and camera noise completely.
   - When hand accelerates: cutoff frequency immediately ramps up, resulting in **zero lag** during fast flicks across the screen.
3. **Scale-Invariant Pinch & Debounce**:
   Fingertip distances are normalized by palm dimensions (wrist-to-knuckle distance). Gestures remain reliable whether your hand is 30 cm or 1.5 m away from the camera, with hysteresis debouncing to prevent click fluttering.
4. **Active Interaction Box (ROI) & Mirroring**:
   Maps a comfortable central interaction zone to 100% of your monitor resolution so you don't get "gorilla arm" fatigue having to reach across the entire camera view.

---

## 🖐️ Gesture Controls

| Gesture | Finger Pose | Action | Behavior |
| :--- | :--- | :--- | :--- |
| **Move Cursor** | Point Index Finger | **Cursor Movement** | Smooth, low-latency tracking via 1€ filter |
| **Left Click** | Pinch Thumb + Index | **Left Click** | Immediate single click with position lock |
| **Double Click** | Quick Double Pinch | **Double Click** | Opens files, folders, and links |
| **Drag & Drop** | Hold Pinch (> 250ms) | **Mouse Down & Move** | Select text, drag windows, move icons |
| **Right Click** | Pinch Thumb + Middle | **Right Click** | Opens context menus |
| **Scroll** | Extend Index & Middle together | **Vertical & Horizontal Scroll** | Move hand up/down or left/right to scroll |
| **Clutch / Pause** | Press `[Space]` | **Pause Tracking** | Rest your hand or reposition your arm |

---

## 🚀 Quick Start

### 1. Requirements & Installation

- **Python 3.9+**
- Webcam (built-in or USB)

```bash
git clone https://github.com/wmVVII/airmouse.git
cd airmouse

# Install dependencies
pip install -r requirements.txt
```

*(Note: The MediaPipe Hand Landmarker model auto-downloads on first launch if not already cached).*

---

### 2. Running the Air Mouse

#### Launch with Interactive HUD (Calibration & Feedback):
```bash
python -m airmouse.main
```

#### Launch in Headless / High-Performance Background Mode:
```bash
python -m airmouse.main --no-gui
```

#### CLI Options:
```
  --camera CAMERA       Camera device index (default: 0)
  --width WIDTH         Camera capture width (e.g. 640 or 1280)
  --height HEIGHT       Camera capture height (e.g. 480 or 720)
  --fps FPS             Camera target FPS (e.g. 60)
  --no-gui              Disable HUD window for minimum CPU usage
  --sensitivity FLOAT   Adjust cursor speed multiplier (default: 1.0)
  --mincutoff FLOAT     1€ filter base cutoff (lower = more tremor suppression)
  --beta FLOAT          1€ filter speed coefficient (higher = zero lag on fast moves)
  --config PATH         Path to custom YAML configuration
```

---

## ⚙️ Configuration (`config/default.yaml`)

You can customize all tracking, sensitivity, and gesture thresholds in `config/default.yaml`:

```yaml
camera:
  camera_id: 0
  width: 640
  height: 480
  target_fps: 60

filter:
  freq: 30.0
  mincutoff: 1.0       # 0.5 - 1.5: Lower cancels hand tremor when still
  beta: 5.0            # 2.0 - 10.0: Higher removes lag during fast sweeps
  dcutoff: 1.0

roi:
  x_min: 0.20          # Left boundary of active interaction box (0.0 - 1.0)
  x_max: 0.80          # Right boundary of active interaction box
  y_min: 0.20          # Top boundary
  y_max: 0.80          # Bottom boundary

gestures:
  pinch_press_thresh: 0.28    # Distance threshold to trigger click
  pinch_release_thresh: 0.40  # Hysteresis threshold to release click
  drag_delay: 0.25            # Seconds holding pinch before drag begins
  double_click_delay: 0.35    # Max seconds between clicks for double click
  scroll_threshold: 0.35      # Max distance between index & middle for scroll
  scroll_speed: 30.0          # Scroll wheel velocity multiplier

gui: true
sensitivity: 1.0
```

---

## 🖥️ Operating System Specific Notes

### Windows
- Works out of the box with `pynput` with zero additional configuration or administrator rights needed.

### macOS
- When running for the first time, macOS will ask for **Accessibility** permissions so the application can move the mouse cursor.
- Navigate to: **System Settings > Privacy & Security > Accessibility** and ensure your Terminal / Python environment is checked.

### Linux
- **X11 / XWayland**: Fully supported out of the box.
- **Wayland**: Some Wayland compositors (like GNOME Wayland) restrict global cursor warping by non-compositor applications. If cursor movement is blocked, run under an X11 session or configure `evdev` / `uinput` permissions.

---

## 🧪 Running the Automated Test Suite

The project includes unit tests covering the 1€ filter mathematics, gesture state transitions, debouncing, and end-to-end synthetic frame pipelines:

```bash
pytest -v
```
