"""Acquire into leased shared memory and optionally clock fake sensors."""

from time import monotonic

from pypylon import pylon

from common.new_structs import PhotoExif
from common.pipeline_stage import PipelineStage
from .camera_buffer import LeasedBuffer
from .camera_setup import setup_camera


class CameraEngine(PipelineStage):
    def __init__(self, configs, logger, registry, stage_name):
        super().__init__(configs, logger, registry, stage_name)
        self.cam = self.capture_connection = None
        self.capture_period = 0

    def setup_cam(self):
        self.cam = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
        )
        self.cam.Open()
        setup_camera(self.cam, self.configs)
        self.next_frame_at = monotonic()

    def mission(self):
        if self.stopped.is_set():
            return True
        if self.cam is None:
            self.setup_cam()
        software_trigger = (
            self.cam.TriggerMode.Value == "On"
            and self.cam.TriggerSource.Value == "Software"
        )
        if software_trigger:
            if self.stopped.wait(max(0, self.next_frame_at - monotonic())):
                return True
            self.next_frame_at = monotonic() + self.capture_period
        try:
            with self.lease_slot_for_writing("tagged_image") as buffer:
                photo = PhotoExif.from_buffer(buffer)
                try:
                    self.cam.SetBufferFactory(LeasedBuffer(buffer))
                    self.cam.StartGrabbingMax(1, pylon.GrabStrategy_OneByOne)
                    if software_trigger:
                        self.cam.WaitForFrameTriggerReady(
                            1000, pylon.TimeoutHandling_ThrowException
                        )
                        self.cam.ExecuteSoftwareTrigger()
                    result = self.cam.RetrieveResult(500, pylon.TimeoutHandling_Return)
                    if result is None:
                        raise TimeoutError("Camera grab timed out")
                    with result:
                        if not result.GrabSucceeded():
                            raise TimeoutError(result.ErrorDescription)
                        if (
                            self.cam.ChunkModeActive.Value
                            and self.cam.ChunkSelector.Value == "CounterValue"
                        ):
                            result.ChunkCounterSelector.Value = (
                                self.cam.CounterSelector.Value
                            )
                            capture_id = result.ChunkCounterValue.Value
                        else:
                            capture_id = result.GetBlockID()
                        photo.ID = capture_id
                        photo.exposure_time, photo.gain = (
                            self.cam.ExposureTime.Value,
                            self.cam.Gain.Value,
                        )
                        photo.gamma, photo.black_level = (
                            self.cam.Gamma.Value,
                            self.cam.BlackLevel.Value,
                        )
                        photo.bsl_color_space = self.cam.BslColorSpace.Value.encode()
                        photo.light_source_preset = (
                            self.cam.BslLightSourcePreset.Value.encode()
                        )
                finally:
                    self.cam.StopGrabbing()
                    self.cam.SetBufferFactory(None)
                    del photo
        except TimeoutError as error:
            self.logger.warning("%s", error)
            return self.stopped.is_set()
        if self.capture_connection is not None:
            self.capture_connection.send(capture_id)
        return self.stopped.is_set()

    def stop(self):
        self.stopped.set()

    def _dump_data(self):
        if self.cam is not None:
            self.cam.Close()
        if self.capture_connection is not None:
            self.capture_connection.close()
