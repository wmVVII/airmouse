from pathlib import Path
import pytest
from airmouse.core.config import AppConfig


def test_config_defaults():
    cfg = AppConfig.load()
    assert cfg.camera.width == 640
    assert cfg.camera.height == 480
    assert cfg.camera.target_fps == 60
    assert cfg.filter.mincutoff == 1.0
    assert cfg.filter.beta == 5.0
    assert cfg.roi.x_min == 0.20
    assert cfg.roi.x_max == 0.80
    assert cfg.gestures.pinch_press_thresh == 0.28


def test_config_save_and_load(tmp_path: Path):
    temp_yaml = tmp_path / "custom_config.yaml"
    cfg = AppConfig()
    cfg.camera.width = 1280
    cfg.camera.height = 720
    cfg.filter.beta = 8.0
    cfg.sensitivity = 1.5

    cfg.save(temp_yaml)
    assert temp_yaml.exists()

    loaded = AppConfig.load(temp_yaml)
    assert loaded.camera.width == 1280
    assert loaded.camera.height == 720
    assert loaded.filter.beta == 8.0
    assert loaded.sensitivity == 1.5
