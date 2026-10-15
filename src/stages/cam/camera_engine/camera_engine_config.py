from core.options import Options

CONFIG = {
    "resolution": [5320, 3032],
    "channels": 1,
    "emulated": Options(
        {"mock": True, "no_mock": False}, default="no_mock"
    ),  # whether the camera is emulated
    "emulation_use_random_data": Options(
        (True, False), alias=("use random data", "use real data"), default=1
    ),
    "mock_metadata_csv": (
        "/home/sft/drone_dataset_yolo_final/val/metadata_pixhawk_bayerrg8.csv"
    ),
    "start_delay": Options(
        range(0, 300), 0
    ),  # delay before starting the camera in seconds
    "acquisition_time": Options(range(0, 60 * 30), 300),  # acquisition time in seconds
    "pixel_format": "BayerRG8",  # pixel format for the camera
    "exposure_mode": "Timed",  # export mode for the camera
    "exposure_auto": "Continuous",  # auto exposure setting
    "auto_gain": "Continuous",  # auto gain setting
    "gamma": 1,  # gamma correction setting
    "black_level": 0,  # black level setting
    "bsl_color_space": "sRgb",  # BSL color space setting
    "balance_white_auto": "Continuous",  # auto white balance setting
    "light_source_preset": "Daylight5000K",  # light source preset setting
    "csv_out_dir": "./data/outs/pictures.csv",  # directory path for CSV output
    "img_out_dir": "./data/outs",  # directory path for image output
    # "saving_mode": "NPY",  # saving mode for the camera
    # "metadata_db": "./data/outs/metadata.csv",  # metadata database settings
}
