import ctypes as ct
from importlib.util import module_from_spec, spec_from_file_location

import pytest
import yaml

from common import config, new_structs
from core.central_registry import CentralRegistry


def load_structs(monkeypatch, tmp_path, settings):
    mission_path = tmp_path / "mission.yaml"
    mission_path.write_text(
        yaml.safe_dump(
            [
                {"sensor_generator": [{"default": []}]},
                {"cam": [{"mock_camera_engine": settings}]},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "MISSION_CONFIG_PATH", mission_path)
    spec = spec_from_file_location("test_producer_structs", new_structs.__file__)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("channels", [1, 3])
def test_selected_image_producer_settings_determine_shared_memory_size(
    monkeypatch, tmp_path, channels
):
    structs = load_structs(monkeypatch, tmp_path, [{"resolution": [8, 6]}, {"channels": channels}])
    assert structs.IMAGE_SHAPE == ((6, 8) if channels == 1 else (6, 8, channels))
    assert structs.PhotoExif.image.size == 8 * 6 * channels
    registry = CentralRegistry()
    registry.register_producer("cam", structs.PhotoExif)
    helper = registry.data_array[structs.PhotoExif]
    try:
        assert helper.cell_size == ct.sizeof(structs.PhotoExif)
        with registry.lease_slot_for_writing(structs.PhotoExif) as view:
            photo = structs.PhotoExif.from_buffer(view)
            photo.ID, photo.image[-1] = 42, 255
            del photo
        assert 42 in registry.entries
    finally:
        helper.close()


def test_image_layout_uses_selected_producer_config_defaults(monkeypatch, tmp_path):
    structs = load_structs(monkeypatch, tmp_path, [])
    assert structs.IMAGE_SHAPE == (3032, 5320)
    assert structs.PhotoExif.image.size == 5320 * 3032
