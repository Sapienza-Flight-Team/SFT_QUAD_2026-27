from unittest.mock import MagicMock

import pytest
import yaml

from common.config import MISSION_CONFIG_PATH, UTILITY_CONFIG_PATH
from common.new_structs import IMAGE_SHAPE, PhotoExif
from core import pipeline as pipeline_module
from core.stage_catalog import read_profile


def test_profile_selects_implementation_and_preserves_settings_values():
    catalog = {
        "camera": {"real": {"module": "real_camera"}, "mock": {"module": "fake_camera"}},
        "viewer": {"default": {"module": "viewer"}},
    }
    profile = [
        {
            "camera": [
                {
                    "mock": [
                        {"rate": 29},
                        {"profile": {"gain": 2}},
                        {"resolution": [640, 480]},
                    ]
                }
            ]
        },
        {"viewer": [{"default": [{"display": False}]}]},
    ]
    selected = read_profile(catalog, profile)
    assert list(selected) == ["camera", "viewer"]
    assert selected["camera"] == {
        "module": "fake_camera",
        "settings": {"rate": 29, "profile": {"gain": 2}, "resolution": [640, 480]},
    }
    assert selected["viewer"]["settings"] == {"display": False}
    selected["camera"]["class"] = object
    assert catalog["camera"]["mock"] == {"module": "fake_camera"}


@pytest.mark.parametrize("settings", [None, []])
def test_profile_without_overrides_supplies_an_empty_settings_dictionary(settings):
    selected = read_profile({"viewer": {"default": {}}}, [{"viewer": [{"default": settings}]}])
    assert selected["viewer"]["settings"] == {}


def test_current_files_build_stages_with_their_own_settings(monkeypatch):
    logger = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(pipeline_module, "SFTLogger", logger)
    mission = yaml.safe_load(MISSION_CONFIG_PATH.read_text(encoding="utf-8"))
    utilities = yaml.safe_load(UTILITY_CONFIG_PATH.read_text(encoding="utf-8"))
    pipeline = pipeline_module.Pipeline(mission, utilities)
    try:
        dummy, camera, sensor, viewer = pipeline.build()
        logger.assert_called_once_with(config=utilities["logger"])
        dummy_options = next(entry["dummy_consumer"][0]["default"]
                             for entry in mission if "dummy_consumer" in entry)
        assert dummy.configs == {key: value for option in dummy_options for key, value in option.items()}
        assert camera.configs["trigger_source"] == "Software" and sensor.options["external_clock"] is True
        assert camera.capture_connection.writable and not camera.capture_connection.readable
        assert sensor.capture_connection.readable and not sensor.capture_connection.writable
        camera.capture_connection.send(42)
        assert sensor.capture_connection.recv() == 42
        assert sensor.options["latitude"] == 45.0
        assert viewer.options["display"] is False
        assert viewer.options["preview_resolution"] == [1064, 606]
        assert all(
            "logger" not in stage.configs and "image" not in stage.configs
            for stage in pipeline.processes
        )
        assert "image" not in utilities
        width, height = camera.configs["resolution"]
        assert IMAGE_SHAPE == (height, width)
        assert PhotoExif.image.size == width * height * camera.configs["channels"]
    finally:
        pipeline.close_registry()
