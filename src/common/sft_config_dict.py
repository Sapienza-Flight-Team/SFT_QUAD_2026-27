from pathlib import Path

import yaml

from stages.common.options import Options


def _load_px2body_matrix_from_yaml() -> list[list[float]]:
    config_path = Path(__file__).resolve().parents[3] / "sft_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config["decision_system"]["geoloc"]["px2body_matrix"]


def _load_px2body_error_angle_deg_from_yaml() -> float:
    config_path = Path(__file__).resolve().parents[3] / "sft_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return float(config["decision_system"]["geoloc"]["px2body_error_angle_deg"])


def _load_real_target_location_from_yaml() -> list[float]:
    config_path = Path(__file__).resolve().parents[3] / "sft_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    target = config["decision_system"]["geoloc"]["real_target_location"]
    return [float(target[0]), float(target[1])]


UTILS = {
    "yaml_path": "sft_config.yaml",
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
    "mock_metadata_csv": (
        "/home/sft/drone_dataset_yolo_final/val/metadata_pixhawk_bayerrg8.csv"
    ),
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
        "focal_length": 16.0,
        "pixel_size": 2.74,
        "image_shape": [5320, 3032],
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
    "instance_splitter": {
        # separa le detection di una stessa classe (con clusters > 1) in
        # istanze distinte prima del clustering, in base alla distanza spaziale
        "enabled": True,
        "min_instance_distance_m": 15.0,  # detection oltre questa soglia = istanza diversa
    },
    # Current 4-class target namespace:
    # 0=GCP_numbered, 1=GCP_non_numbered, 2=bullseye, 3=X.
    "targets": {
        0: {"name": "GCP_numbered", "clusters": 1, "clusterize": True},
        1: {"name": "GCP_non_numbered", "clusters": 1, "clusterize": True},
        2: {"name": "BULLSEYE", "clusters": 0, "clusterize": False},
        3: {"name": "X", "clusters": 0, "clusterize": False},
    },
}

PGRCField_MAPPING = {
    "tl": [38.851917, -76.697570],
    "bl": [38.852463, -76.698960],
    "br": [38.849820, -76.700667],
    "tr": [38.849270, -76.699286],
}
PGRCField_AIRDROP = {
    "tl": [38.851414, -76.699265],
    "bl": [38.851502, -76.699529],
    "br": [38.850246, -76.700228],
    "tr": [38.850158, -76.699969],
}
MARYLAND_MAPPING_RUNWAY1 = {
    "tl": [38.316639, -76.552330],
    "bl": [38.315460, -76.552653],
    "br": [38.314816, -76.548947],
    "tr": [38.316016, -76.548600],
}
MARYLAND_MAPPING_RUNWAY2 = {
    "tl": [38.315873, -76.547611],
    "bl": [38.314669, -76.547987],
    "br": [38.314008, -76.544237],
    "tr": [38.315208, -76.543840],
}
MARYLAND_AIRDROP_RUNWAY1 = {
    "tl": [38.315895, -76.552519],
    "bl": [38.315683, -76.552586],
    "br": [38.315386, -76.550875],
    "tr": [38.315607, -76.550800],
}
MARYLAND_AIRDROP_RUNWAY2 = {
    "bl": [38.314228, -76.544156],
    "tl": [38.314441, -76.544081],
    "tr": [38.314731, -76.545792],
    "br": [38.314529, -76.545859],
}
PONTE_GALERIA_MAPPING = {
    "tl": [41.833120, 12.334739],
    "tr": [41.830228, 12.335866],
    "bl": [41.832775, 12.333153],
    "br": [41.829912, 12.334282],
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
#     "wind_h": 50.0,  # wind direction in degrees
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
#             MARYLAND_AIRDROP_RUNWAY1,
#             MARYLAND_AIRDROP_RUNWAY2,
#             PGRCField_AIRDROP,
#         ),
#         alias=(
#             "labs coords",
#             "ponte galeria",
#             "whole world",
#             "maryland airdrop runway 1",
#             "maryland airdrop runway 2",
#             "PGRC field airdrop",
#         ),
#         default=2,
#     ),  # coordinates of the manouver area
# }


MOCK_PIPELINE = {
    "stages": [],  # list of pipeline stages
    "enabled": False,  # whether the pipeline is active
}

TEENSY_MOCK_IP = "127.0.0.1"
TEENSY_MOCK_PORT = 6969

TEENSY_IP = "192.168.1.2"
TEENSY_PORT = 80

TEENSY = {
    "ip_addr": Options(
        {"mock": TEENSY_MOCK_IP, "no_mock": TEENSY_IP}, default="no_mock"
    ),  # IP address of the teensy
    "port": Options(
        {"mock": TEENSY_MOCK_PORT, "no_mock": TEENSY_PORT}, default="no_mock"
    ),  # port of the teensy
    "timeout": 15,  # timeout for the socket
    "sample_size": 3,  # Samples of past gps positions to keep in the queue
}

MAVLINK_MOCK_PORT = "/dev/pts/13"
MAVLINK_PORT = "/dev/ttyACM0"
MAVLINK_MOCK_CONNECTION = "udpin:127.0.0.1:14550"
MAVLINK_CONNECTION_NO_MOCK = "/dev/ttyACM0"

MAVLINK = {
    "serial_port": Options(
        {"mock": MAVLINK_MOCK_PORT, "no_mock": MAVLINK_PORT}, default="no_mock"
    ),
    "connection": Options(
        {"mock": MAVLINK_MOCK_CONNECTION, "no_mock": MAVLINK_CONNECTION_NO_MOCK},
        default="no_mock",
    ),
    "baud": 57600,
    "demo": False,
    "require_vground_for_ready": Options(
        {"mock": False, "no_mock": True}, default="no_mock"
    ),
    "timeout": 15,
    "sample_size": 3,
}

AIRDROP_BOUND = Options(
    (
        LABS_COORDS,
        PONTE_GALERIA,
        ZERO_ZERO_COORDS,
        MARYLAND_AIRDROP_RUNWAY1,
        MARYLAND_AIRDROP_RUNWAY2,
        PGRCField_AIRDROP,
    ),
    alias=(
        "labs coords",
        "ponte galeria",
        "whole world",
        "maryland airdrop runway 1",
        "maryland airdrop runway 2",
        "PGRC field airdrop",
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

GMM = {
    "gmm_kernel_size": 5,
    "gmm_k": 2,
    "gmm_n_iter": 10,
    "gmm_min_increment": 0.01,
    "perc_flattened_img": 0.5,
}

INFERENCE = {
    "downscale_factor": 1.0,  # Set your default downscale factor here
    "imgsz": [2656, 1504],
    "yolo_weights": "models/yolo/best.pt",
    "classes": [0, 1, 2, 3],
    "yolo_class_map": {
        0: "GCP_numbered",
        1: "GCP_non_numbered",
        2: "BULLSEYE",
        3: "X",
    },
    "blob_detector": BLOBDETECTOR,
    "gmm": GMM,
}

PREPROCESSOR = {
    "target_resolution": [2656, 1504],
}

BBOX_LOGGING = {
    "enabled": True,
    "db_path": None,
    "busy_timeout_ms": 1000,
}

# Combine all subsystem dictionaries into the master configuration dictionary
STFCONFIGDICT = {
    "utils": UTILS,
    "logger": LOGGER,
    "camera": CAMERA,
    "preprocessor": PREPROCESSOR,
    "picture_saver": PICTURE_SAVER,
    "decision_system": DECISION_SYSTEM,
    # "fall_model": FALL_MODEL,
    "mock_pipeline": MOCK_PIPELINE,
    "teensy": TEENSY,
    "mavlink": MAVLINK,
    "airdrop_bound": AIRDROP_BOUND,
    "inference_engine": INFERENCE,
    "bbox_logging": BBOX_LOGGING,
}
