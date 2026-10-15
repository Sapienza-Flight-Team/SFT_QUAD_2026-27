import pypylon.pylon as pylon
import numpy as np
import cv2
from common.config import SFTConfig

cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
cam.Open()


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


def compute_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)  # Apply Laplacian filter
    return np.var(laplacian)


def compute_sobel_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X gradient
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y gradient
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
    variance = np.var(gray)  # Compute variance of pixel intensities
    return np.mean(sobel_magnitude) + variance  # Combine Sobel and variance


# 9000 sobv 0 lap
# 4500 150 focus
laplacian_var = 0
while cam.IsGrabbing():
    result = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    img = result.Array

    ar = result.Width / result.Height
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)

    x = int(imgrgb.shape[0] * 0.05)
    y = int(imgrgb.shape[1] * 0.05)
    corner = (imgrgb.shape[0] // 2 - x // 2, imgrgb.shape[1] // 2 - y // 2)
    img_lap = imgrgb[
        corner[0] : corner[0] + x,
        corner[1] : corner[1] + y,
    ]

    imgrgb = cv2.resize(imgrgb, (3000, int(3000 / ar)), interpolation=cv2.INTER_LINEAR)

    # Compute Laplacian variance
    # Compute Sobel variance
    # sobel_var = compute_sobel_variance(imgrgb)

    # crop a ROI for the laplacian

    # show the ROI
    cv2.imshow("ROI", img_lap)

    # Compute running average
    laplacian_var = 0.4 * laplacian_var + 0.6 * compute_laplacian(img_lap)

    # Normalize the variance
    # sobel_var = (sobel_var - 4500) / (12000 - 4500)
    laplacian_var = (laplacian_var - 0) / (250 - 0)
    # sobel_var = max(1-sobel_var, 0)
    laplacian_var = min(laplacian_var, 1)

    # print(f"Sobel variance: {sobel_var}")
    # print(f"Laplacian variance: {laplacian_var}")
    # compute running average

    bar_x = 50  # Starting x-coordinate
    bar_y_sobel = 50  # Starting y-coordinate for Sobel bar
    bar_y_laplacian = 100  # Starting y-coordinate for Laplacian bar
    bar_width = 500  # Total width of the bar
    bar_height = 20  # Height of the bar

    # Draw Sobel progress bar
    # cv2.rectangle(imgrgb, (bar_x, bar_y_sobel), (bar_x + bar_width, bar_y_sobel + bar_height), (200, 200, 200), -1)  # Background
    # cv2.rectangle(imgrgb, (bar_x, bar_y_sobel), (bar_x + int(bar_width * sobel_var), bar_y_sobel + bar_height), (0, 255, 0), -1)  # Progress

    # Draw Laplacian progress bar
    cv2.rectangle(
        imgrgb,
        (bar_x, bar_y_laplacian),
        (bar_x + bar_width, bar_y_laplacian + bar_height),
        (200, 200, 200),
        -1,
    )  # Background
    cv2.rectangle(
        imgrgb,
        (bar_x, bar_y_laplacian),
        (bar_x + int(bar_width * laplacian_var), bar_y_laplacian + bar_height),
        (255, 0, 0),
        -1,
    )  # Progress

    # Add text labels for the progress bars
    cv2.putText(
        imgrgb,
        "Laplacian Variance",
        (bar_x, bar_y_laplacian - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
    )

    print(f"Laplacian variance: {laplacian_var}")
    print(f"Sobel Variance: ", compute_sobel_variance(imgrgb))

    cv2.imshow("test_image.png", imgrgb)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break
    result.Release()

cam.StopGrabbing()
cam.Close()
