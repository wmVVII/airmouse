"""
Configuration data classes and YAML loader for Air Mouse settings.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    camera_id: int = 0
    width: int = 640
    height: int = 480
    target_fps: int = 60


@dataclass
class FilterConfig:
    freq: float = 30.0
    mincutoff: float = 1.0
    beta: float = 5.0
    dcutoff: float = 1.0


@dataclass
class ROIConfig:
    x_min: float = 0.20
    x_max: float = 0.80
    y_min: float = 0.20
    y_max: float = 0.80


@dataclass
class GestureConfig:
    pinch_press_thresh: float = 0.28
    pinch_release_thresh: float = 0.40
    drag_delay: float = 0.25
    double_click_delay: float = 0.35
    scroll_threshold: float = 0.35
    scroll_speed: float = 30.0


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    gui: bool = True
    model_path: Optional[str] = None
    sensitivity: float = 1.0

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "AppConfig":
        """Loads configuration from YAML file or returns defaults if not found."""
        config = cls()
        if config_path is None:
            # Check default path
            default_paths = [
                Path("config/default.yaml"),
                Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml",
            ]
            for p in default_paths:
                if p.exists():
                    config_path = p
                    break

        if config_path is not None and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    raw_data: Dict[str, Any] = yaml.safe_load(f) or {}

                if "camera" in raw_data:
                    config.camera = CameraConfig(**raw_data["camera"])
                if "filter" in raw_data:
                    config.filter = FilterConfig(**raw_data["filter"])
                if "roi" in raw_data:
                    config.roi = ROIConfig(**raw_data["roi"])
                if "gestures" in raw_data:
                    config.gestures = GestureConfig(**raw_data["gestures"])
                if "gui" in raw_data:
                    config.gui = bool(raw_data["gui"])
                if "model_path" in raw_data:
                    config.model_path = raw_data["model_path"]
                if "sensitivity" in raw_data:
                    config.sensitivity = float(raw_data["sensitivity"])

                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to parse config file at {config_path}: {e}. Using defaults.")

        return config

    def save(self, config_path: str | Path) -> None:
        """Saves current configuration to a YAML file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "camera": asdict(self.camera),
            "filter": asdict(self.filter),
            "roi": asdict(self.roi),
            "gestures": asdict(self.gestures),
            "gui": self.gui,
            "model_path": self.model_path,
            "sensitivity": self.sensitivity,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {path}")
