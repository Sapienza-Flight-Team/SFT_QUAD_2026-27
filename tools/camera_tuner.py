import cv2
import pypylon.pylon as pylon
from common.config import SFTConfig

cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
cam.Open()

RESIZE = 0.5


def setup_cam(cam, config) -> None:
    """setup_cam
    Deferred initialization of the camera, this is done to avoid
    the camera being initialized before the process is spawned,
    resulting in incompatible address spaces.
    """

    cam.Gamma = config.config_dict["camera"]["gamma"]
    cam.BslColorSpace = config.config_dict["camera"]["bsl_color_space"]
    cam.BslLightSourcePreset = config.config_dict["camera"]["light_source_preset"]
    cam.BlackLevel = config.config_dict["camera"]["black_level"]
    cam.BalanceWhiteAuto = config.config_dict["camera"]["balance_white_auto"]
    cam.GainAuto = config.config_dict["camera"]["auto_gain"]
    cam.AutoGainLowerLimit = config.config_dict["camera"]["profile"]["min_auto_gain"]
    cam.AutoGainUpperLimit = config.config_dict["camera"]["profile"]["max_auto_gain"]
    cam.PixelFormat = config.config_dict["camera"]["pixel_format"]
    cam.ExposureMode = config.config_dict["camera"]["exposure_mode"]
    cam.ExposureAuto = config.config_dict["camera"]["exposure_auto"]
    cam.AutoExposureTimeLowerLimit = config.config_dict["camera"]["profile"][
        "min_exposure_time"
    ]
    cam.AutoExposureTimeUpperLimit = config.config_dict["camera"]["profile"][
        "max_exposure_time"
    ]
    cam.AcquisitionFrameRateEnable = True
    cam.AcquisitionFrameRate = 10.0
    return


setup_cam(cam, SFTConfig("sft_config.yaml"))
cam.StartGrabbing()

while cam.IsGrabbing():
    grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        image = grabResult.Array
        image = cv2.cvtColor(image, cv2.COLOR_BayerBG2BGR)
        h = image.shape[0]
        w = image.shape[1]
        image = cv2.resize(image, (int(w * RESIZE), int(h * RESIZE)))
        cv2.imshow("image", image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            cam.StopGrabbing()
            break
        elif key == ord("r"):
            cam.StopGrabbing()
            setup_cam(cam, SFTConfig("sft_config.yaml"))
            cam.StartGrabbing()
    grabResult.Release()
cam.Close()
cam.DestroyInstance()
