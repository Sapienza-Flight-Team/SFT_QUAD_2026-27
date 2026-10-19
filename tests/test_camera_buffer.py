import ctypes as ct
import os
import subprocess
import sys

import pytest

from common.new_structs import PhotoExif
from stages.cam.camera_engine.camera_buffer import LeasedBuffer


def test_factory_checks_capacity_and_never_reuses_a_lease_for_two_buffers():
    lease = memoryview(bytearray(ct.sizeof(PhotoExif)))
    factory = LeasedBuffer(lease)
    with pytest.raises(ValueError, match="single leased buffer"):
        factory.AllocateBuffer(len(factory.view) + 1)
    assert factory.AllocateBuffer(len(factory.view)) == ct.addressof(
        ct.c_uint8.from_buffer(lease, PhotoExif.image.offset)
    )
    with pytest.raises(ValueError, match="single leased buffer"):
        factory.AllocateBuffer(1)
    factory.FreeBuffer(None, None)
    factory.OnReleased()
    lease.release()


def test_pylon_emulator_writes_to_the_supplied_lease_address():
    script = """
import ctypes as ct
from pypylon import pylon
from common.new_structs import PhotoExif
from stages.cam.camera_engine.camera_buffer import LeasedBuffer

factory = pylon.TlFactory.GetInstance()
info = pylon.DeviceInfo()
info.SetDeviceClass("BaslerCamEmu")
cam = pylon.InstantCamera(factory.CreateFirstDevice(info))
cam.Open()
cam.PixelFormat.Value = "Mono8"
cam.Width.Value, cam.Height.Value = 64, 48
cam.TriggerSelector.Value = "FrameStart"
cam.TriggerMode.Value, cam.TriggerSource.Value = "On", "Software"
lease = memoryview(bytearray(ct.sizeof(PhotoExif)))
try:
    cam.SetBufferFactory(LeasedBuffer(lease))
    cam.StartGrabbingMax(1, pylon.GrabStrategy_OneByOne)
    cam.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_ThrowException)
    cam.ExecuteSoftwareTrigger()
    with cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as result:
        assert result.GrabSucceeded()
        with result.GetArrayZeroCopy() as pixels:
            assert pixels.ctypes.data == ct.addressof(ct.c_uint8.from_buffer(lease, PhotoExif.image.offset))
            pixels[0, 0] = 137
            assert lease[PhotoExif.image.offset] == 137
finally:
    cam.StopGrabbing()
    cam.SetBufferFactory(None)
    cam.Close()
    lease.release()
print("SDK wrote directly into supplied memory")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYLON_CAMEMU": "1"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SDK wrote directly into supplied memory" in result.stdout
