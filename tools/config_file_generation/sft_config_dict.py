from pathlib import Path

import yaml

from core.options import Options


def _load_px2body_matrix_from_yaml() -> list[list[float]]:
    config_path = Path(__file__).resolve().parents[2] / "sft_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config["decision_system"]["geoloc"]["px2body_matrix"]


def _load_px2body_error_angle_deg_from_yaml() -> float:
    config_path = Path(__file__).resolve().parents[2] / "sft_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return float(config["decision_system"]["geoloc"]["px2body_error_angle_deg"])


def _load_real_target_location_from_yaml() -> list[float]:
    config_path = Path(__file__).resolve().parents[2] / "sft_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    target = config["decision_system"]["geoloc"]["real_target_location"]
    return [float(target[0]), float(target[1])]


UTILS = {
    "yaml_path": "sft_config.yaml",
    "mission_yaml": "mission_config.yaml",
    "height_threshold": 10,
}

LOGGER = {
    "log_level": Options(("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), default=1),
    "log_file": "/tmp/sft_cv.log",
    "log_format": "[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    "log_date_format": "%Y-%m-%d %H:%M:%S",
    "log_console": Options((True, False), default=0),
}

AUTO = {
    "min_exposure_time": None,  # minimum exposure time
    "max_exposure_time": None,  # maximum exposure time
    "min_gain": None,  # minimum gain value
    "max_gain": None,  # maximum gain value
}

PROFILE = {
    "min_exposure_time": Options(
        range(100, 5000), default=100
    ),  # TODO ONE RANGE IS DEPENDENT ON THE OTHER
    "max_exposure_time": Options(range(100, 10000), default=2000),
    "min_auto_gain": Options(range(0, 30), default=0),
    "max_auto_gain": Options(range(0, 30), default=5),
}

CAMERA = {
    "emulated": Options(
        {"mock": True, "no_mock": False}, default="no_mock"
    ),  # whether the camera is emulated
    "emulation_use_random_data": Options(
        (True, False), alias=("use random data", "use real data"), default=1
    ),
    "mock_metadata_csv": Options(
        "/home/sft/drone_dataset_yolo_final/val/metadata_pixhawk_bayerrg8.csv",
        default="",
    ),
    "sensor_width_mm": 14.58,
    "sensor_height_mm": 8.31,
    "focal_length_mm": 16.0,
    "frontal_overlap": 0.8,
    "side_overlap": 0.2,
    "resolution": [5320, 3032],
    "profile": PROFILE,  # active camera profile
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
    "profile": PROFILE,  # camera profile settings
    "csv_out_dir": "./data/outs/pictures.csv",  # directory path for CSV output
    "img_out_dir": "./data/outs",  # directory path for image output
    # "saving_mode": "NPY",  # saving mode for the camera
    # "metadata_db": "./data/outs/metadata.csv",  # metadata database settings
    "sensor_width_mm": 14.58,
    "sensor_height_mm": 8.31,
    "focal_length_mm": 16.0,
    "frontal_overlap": 0.8,
    "side_overlap": 0.2,
}

PICTURE_SAVER = {
    "saving_mode": "NPY",
    "metadata_db": "./data/outs/metadata.csv",
}

DECISION_SYSTEM = {
    "root": "./src/sft_cv/decision_system/.tmp",  # root where the decision system will save its activities
    "debug": Options((True, False), default=False),  # debugger of decision system
    "mock": {
        "metadata_csv": "./data/synth/pontegaleria/metadata.csv",  # name of the metadata file
    },
    "geoloc": {
        "image_shape": [5320, 3032],
        "focal_length": 16.0,
        "pixel_size": 2.74,
        "px2body_matrix": _load_px2body_matrix_from_yaml(),
        "px2body_error_angle_deg": _load_px2body_error_angle_deg_from_yaml(),
        "real_target_location": _load_real_target_location_from_yaml(),
        "monte_carlo": 1000,
        "p_value": 70.0,  # confidenza data alle caratteristiche qui di seguito
        "rp_error": 3.0,  # errore della gimbal
        "rp_maxerror": 5.0,
        "yaw_error": 0.75,  # errore dell'ahrs
        "yaw_maxerror": 3.0,
        "altitude_error": 2.0,  # errore del barometro
        "altitude_maxerror": 5.0,
        "ne_error": 1.0,  # errore della posizione gps
        "ne_maxerror": 2.0,
        "manifold_center": [38.315748, -76.551709],  # centro del manifold
    },
    "class_handler": {
        "classes": [0, 1, 2, 3],
    },
    "decision": {
        "horizon": 0.99,  # aumentare questo valore per allargare l'orizzonte dei centroidi
        "filter": 0.60,  # ridurre questo valore per aumentare il numero di centroidi
    },
}


PONTE_GALERIA = {
    "tl": [41.83216329770343, 12.332750914195465],
    "tr": [41.832155297115676, 12.333641354892045],
    "bl": [41.83065217380948, 12.33301449995502],
    "br": [41.830735093929135, 12.334065375236168],
}
LABS_COORDS = {
    "tl": [41.972630972528584, 12.612894366949762],
    "tr": [41.99079140740237, 12.311587886238678],
    "bl": [41.78709875524665, 12.617780417988321],
    "br": [41.800455642764184, 12.363705763983297],
}
ZERO_ZERO_COORDS = {
    "tl": [-90.0, -180.0],
    "tr": [-90.0, 180.0],
    "bl": [90.0, 180.0],
    "br": [90.0, -180.0],
}

# FALL_MODEL = {
#     "t_freefall": 1.6,  # opening time in seconds
#     "wind_v": 4.0,  # wind speed in m/s
#     "wind_h": 50.0,  # wind heading in degrees
#     "cd": 0.9,
#     "clockwise": Options(
#         (0, 1), alias=("anticlockwise", "clockwise"), default=0
#     ),  # 0 is anticlockwise, 1 is clockwise
#     "default_v_mod": 18,  # expected value of vmod, it won't be used if gps data has non-zero vmod value
#     "manouver_bound": Options(
#         (
#             LABS_COORDS,
#             PONTE_GALERIA,
#             ZERO_ZERO_COORDS,
#         ),
#         alias=(
#             "labs coords",
#             "ponte galeria",
#             "whole world",
#         ),
#         default=2,
#     ),  # coordinates of the manouver area
# }

MOCK_PIPELINE = {
    "stages": [],  # list of pipeline stages
    "enabled": False,  # whether the pipeline is active
}

MAVLINK_MOCK_PORT = "/dev/pts/13"

MAVLINK_PORT = "/dev/ttyACM0"

MAVLINK = {
    "serial_port": Options(
        {"mock": MAVLINK_MOCK_PORT, "no_mock": MAVLINK_PORT}, default="no_mock"
    ),  # serial or virtual port path for the MAVLink source
    "connection": Options(
        {"mock": "udpin:0.0.0.0:14540", "no_mock": "/dev/ttyACM0"},
        default="no_mock",
    ),
    "baud": 57600,
    "demo": False,
    "require_vground_for_ready": Options(
        {"mock": False, "no_mock": True}, default="no_mock"
    ),
    "timeout": 15,  # timeout for the socket
    "sample_size": 3,  # Samples of past gps positions to keep in the queue
    "mock_type": Options(
        ("simulated", "fake"),
        alias=("simulated (px4_sitl)", "fake (.csv replay)"),
        default=0,
    ),
    "unreal_host_ip": Options("127.0.0.1", default="127.0.0.1"),
    "unreal_video_stream_port": Options(range(1024, 65535), default=9999),
    "unreal_trigger_port": Options(range(1024, 65535), default=3001),
}

AIRDROP_BOUND = Options(
    (
        LABS_COORDS,
        PONTE_GALERIA,
        ZERO_ZERO_COORDS,
    ),
    alias=(
        "labs coords",
        "ponte galeria",
        "whole world",
    ),
    default=2,
)
BLOBDETECTOR = {
    "filterByCircularity": False,
    "filterByColor": False,
    "filterByArea": True,
    "filterByInertia": False,
    "filterByConvexity": False,
    "thresholdStep": 5,
    "minThreshold": 50,
    "maxThreshold": 255,
    "minRepeatability": 1,
    "minDistBetweenBlobs": 10.0,
    "minArea": 33,
}

INFERENCE = {
    "images_per_batch": Options(range(1, 21), default=1),
    "imgsz": [2656, 1504],
    "yolo_weights": "models/yolo/best.pt",
    "classes": [0, 1, 2, 3],
    "yolo_class_map": {
        0: "GCP_numbered",
        1: "GCP_non_numbered",
        2: "BULLSEYE",
        3: "X",
    },
}

PREPROCESSOR = {
    "target_resolution": [2656, 1504],
}

TARGET_SIMULATOR = {
    "blend_alpha": 1.0,
    "unlink_input_shm": True,
    "export_kml": True,
    "kml_output_path": "./data/outs/target_simulator_targets.kml",
    "random_seed": None,
}

# Combine all subsystem dictionaries into the master configuration dictionary
STFCONFIGDICT = {
    "utils": UTILS,
    "logger": LOGGER,
    "camera": CAMERA,
    "preprocessor": PREPROCESSOR,
    "picture_saver": PICTURE_SAVER,
    "decision_system": DECISION_SYSTEM,
    "target_simulator": TARGET_SIMULATOR,
    # "fall_model": FALL_MODEL,
    "mock_pipeline": MOCK_PIPELINE,
    "mavlink": MAVLINK,
    "airdrop_bound": AIRDROP_BOUND,
    "inference_engine": INFERENCE,
}
