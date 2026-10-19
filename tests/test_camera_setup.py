from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from core.options import Options
from core.stage_catalog import build_stage_catalog, read_profile
from stages.cam.camera_engine.camera_engine_config import CONFIG
from stages.cam.camera_engine.camera_setup import setup_camera


class CameraNode:
    Min = 0
    Max = 4294967295

    def __init__(self, camera, name, value=0):
        self.camera, self.name, self.value = camera, name, value

    @property
    def Value(self):
        return self.value

    @Value.setter
    def Value(self, value):
        self.camera.validate(self.name, value)
        self.camera.writes.append((self.name, value))
        self.value = value


class FakeCamera:
    """Check the camera dependencies that require ordered property assignments."""

    def __init__(self):
        names = (
            "PixelFormat OffsetX OffsetY Width Height ReverseX ReverseY AcquisitionMode "
            "TriggerSelector TriggerMode TriggerSource TriggerActivation TriggerDelay "
            "ExposureMode ExposureAuto ExposureTime AutoExposureTimeLowerLimit "
            "AutoExposureTimeUpperLimit GainAuto Gain AutoGainLowerLimit AutoGainUpperLimit "
            "AutoTargetBrightness AutoFunctionProfile BalanceWhiteAuto BslLightSourcePreset "
            "Gamma BlackLevel BslColorSpace AcquisitionFrameRateEnable AcquisitionFrameRate "
            "LineSelector LineMode LineSource LineInverter MaxNumBuffer OutputQueueSize "
            "ChunkModeActive ChunkSelector ChunkEnable CounterSelector CounterEventSource "
            "CounterTriggerSource CounterResetSource CounterDuration SoftwareSignalSelector"
        ).split()
        self.nodes = {name: CameraNode(self, name) for name in names}
        initial = {
            "Width": 640,
            "Height": 480,
            "OffsetX": 100,
            "OffsetY": 100,
            "ExposureAuto": "Continuous",
            "GainAuto": "Continuous",
            "AutoExposureTimeLowerLimit": 4000,
            "AutoExposureTimeUpperLimit": 5000,
            "AutoGainLowerLimit": 10,
            "AutoGainUpperLimit": 20,
            "MaxNumBuffer": 10,
            "OutputQueueSize": 10,
            "LineMode": "Output",
        }
        for name, value in initial.items():
            self.nodes[name].value = value
        self.writes = []
        self.CounterReset, self.SoftwareSignalPulse = Mock(), Mock()

    def __getattr__(self, name):
        if name in self.nodes:
            return self.nodes[name]
        raise AttributeError(name)

    def value(self, name):
        return self.nodes[name].value

    def validate(self, name, value):
        for dimension, offset, maximum in (
            ("Width", "OffsetX", 5320),
            ("Height", "OffsetY", 3032),
        ):
            if name == dimension and value + self.value(offset) > maximum:
                raise ValueError("ROI exceeds sensor")
            if name == offset and value + self.value(dimension) > maximum:
                raise ValueError("ROI exceeds sensor")
        for manual, auto in (("ExposureTime", "ExposureAuto"), ("Gain", "GainAuto")):
            if name == manual and self.value(auto) != "Off":
                raise ValueError("Manual control is locked by auto mode")
        for lower, upper in (
            ("AutoExposureTimeLowerLimit", "AutoExposureTimeUpperLimit"),
            ("AutoGainLowerLimit", "AutoGainUpperLimit"),
        ):
            if name == lower and value > self.value(upper):
                raise ValueError("Lower limit exceeds upper limit")
            if name == upper and value < self.value(lower):
                raise ValueError("Upper limit is below lower limit")
        if name == "MaxNumBuffer" and value < self.value("OutputQueueSize"):
            raise ValueError("Buffer count is below queue size")
        if name == "OutputQueueSize" and value > self.value("MaxNumBuffer"):
            raise ValueError("Queue size exceeds buffer count")
        if name == "TriggerActivation" and self.value("TriggerSource") == "Software":
            raise ValueError("Software triggers have no edge activation")
        if name == "BslColorSpace" and value not in ("sRGB", "RGB"):
            raise ValueError("Invalid color space enumeration")


@pytest.fixture
def config():
    values = {
        name: value.get_default() if isinstance(value, Options) else value
        for name, value in CONFIG.items()
    }
    values["profile"] = {
        name: value.get_default() for name, value in CONFIG["profile"].items()
    }
    return values


def test_gui_yaml_settings_are_passed_to_camera(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QComboBox, QLineEdit
    from tools.config_file_generation import gui

    app = QApplication.instance() or QApplication([])
    catalog = build_stage_catalog()
    editor = gui.ConfigEditor(
        {name: catalog[name] for name in ("cam", "sensor_generator")}
    )
    selector, forms = editor.stages["cam"]
    selector.setCurrentIndex(selector.findData("camera_engine"))
    form = forms.currentWidget()
    form.findChild(QComboBox, "exposure_auto").setCurrentText("Off")
    form.findChild(QComboBox, "auto_gain").setCurrentText("Off")
    form.findChild(QLineEdit, "exposure_time").setText("350.5")
    form.findChild(QLineEdit, "gain").setText("4.5")
    form.findChild(QComboBox, "trigger_source").setCurrentText("Line3")
    form.findChild(QLineEdit, "trigger_delay").setText("12.5")
    sensor_selector, sensor_forms = editor.stages["sensor_generator"]
    sensor_selector.setCurrentIndex(sensor_selector.findData("default"))
    sensor_forms.currentWidget().findChild(QComboBox, "external_clock").setCurrentText(
        "True"
    )
    path = tmp_path / "mission_config.yaml"
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
    editor.save()

    # Use the same flattening as Pipeline.build(), rather than loading in setup.
    settings = read_profile(catalog, yaml.safe_load(path.read_text()))["cam"][
        "settings"
    ]
    cam = FakeCamera()
    setup_camera(cam, settings)
    assert cam.value("ExposureTime") == 350.5
    assert cam.value("Gain") == 4.5
    assert cam.value("TriggerSource") == "Line3"
    assert cam.value("TriggerDelay") == 12.5
    assert cam.value("BslColorSpace") == "sRGB"
    assert cam.value("ChunkSelector") == "CounterValue"
    assert (
        read_profile(catalog, yaml.safe_load(path.read_text()))["sensor_generator"][
            "settings"
        ]["external_clock"]
        is True
    )
    editor.close()
    app.processEvents()


def test_auto_limits_roi_and_buffers_can_change_from_previous_settings(config):
    config.update(max_num_buffer=4, output_queue_size=2)
    cam = FakeCamera()
    setup_camera(cam, config)
    assert cam.value("AutoExposureTimeLowerLimit") == 100
    assert cam.value("AutoExposureTimeUpperLimit") == 2000
    assert cam.value("AutoGainLowerLimit") == 0
    assert cam.value("AutoGainUpperLimit") == 5
    assert cam.value("ExposureAuto") == cam.value("GainAuto") == "Continuous"
    assert cam.value("Width") == 5320 and cam.value("Height") == 3032
    assert cam.value("MaxNumBuffer") == 4 and cam.value("OutputQueueSize") == 2


def test_disabled_features_skip_dependent_properties(config):
    config.update(
        trigger_mode="Off",
        line_mode="Input",
        line_selector="Line3",
        chunk_mode_active=False,
        acquisition_frame_rate_enable=True,
        acquisition_frame_rate=7.5,
    )
    cam = FakeCamera()
    setup_camera(cam, config)
    written = {name for name, value in cam.writes}
    assert not written.intersection(
        {
            "TriggerSource",
            "TriggerActivation",
            "TriggerDelay",
            "LineSource",
            "ChunkSelector",
            "ChunkEnable",
        }
    )
    assert cam.value("AcquisitionFrameRate") == 7.5


@pytest.mark.parametrize(
    "source, exposure", [("Software", "Timed"), ("Line1", "TriggerWidth")]
)
def test_trigger_modes_skip_inapplicable_settings(config, source, exposure):
    config.update(trigger_source=source, exposure_mode=exposure, exposure_auto="Off")
    cam = FakeCamera()
    setup_camera(cam, config)
    assert cam.value("TriggerSource") == source
    assert cam.value("ExposureMode") == exposure
    if source == "Software":
        assert not any(name == "TriggerActivation" for name, value in cam.writes)
    if exposure == "TriggerWidth":
        assert not any(
            name.startswith("AutoExposureTime") or name == "ExposureTime"
            for name, value in cam.writes
        )


def test_camera_errors_propagate_without_wrapping(config):
    config["bsl_color_space"] = "invalid"
    with pytest.raises(ValueError, match="^Invalid color space enumeration$"):
        setup_camera(FakeCamera(), config)


def test_stage_initializes_camera_and_resets_its_counter(monkeypatch, config):
    from stages.cam.camera_engine import camera_engine_stage

    cam = FakeCamera()
    cam.Open = Mock()
    stage = SimpleNamespace(configs=config)
    monkeypatch.setattr(camera_engine_stage.pylon, "TlFactory", Mock())
    monkeypatch.setattr(
        camera_engine_stage.pylon, "InstantCamera", Mock(return_value=cam)
    )
    setup = Mock(wraps=setup_camera)
    monkeypatch.setattr(camera_engine_stage, "setup_camera", setup)
    camera_engine_stage.CameraEngine.setup_cam(stage)
    setup.assert_called_once_with(cam, config)
    cam.Open.assert_called_once_with()
    cam.CounterReset.Execute.assert_called_once_with()
    assert cam.CounterSelector.Value == "Counter2"
    assert cam.CounterEventSource.Value == "FrameStart"
    assert cam.CounterTriggerSource.Value == "SoftwareSignal1"
    cam.SoftwareSignalPulse.Execute.assert_called_once_with()
