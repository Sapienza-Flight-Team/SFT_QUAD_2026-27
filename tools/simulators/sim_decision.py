from .base import SensorsGenerator
from .base import DecisionGenerator
import numpy as np

generator = SensorsGenerator(
    gps_noise=2,
    barometer_noise=4,
    compass_noise=3,
    gyroscope_noise=1,
    data_path="./data/decision_system/Maryland1",
    maps_path="./tools/simulators/assets/AirDrop/Maryland1GEarth.yaml",
    camera_path="./tools/simulators/assets/camera.yaml",
    time_step=0.1,
    lines_csv_path="./tools/simulators/assets/AirDrop/Maryland1GEarth.csv",
    ideal_speed=15.0,
    nominal_speed_noise=4.0,
    nominal_trajectory_brownianNoise=3.0,
    nominal_perspective_noise=5,
    nominal_height_brownianNoise=1.0,
    nominal_yaw_brownianNoise=2.0,
)

decision_generator = DecisionGenerator(
    "./tools/simulators/assets/AirDrop/Maryland1GEarth.png",
    generator,
    good_targets=4,
    intra_distance=15.24,
    n_classes=15,
    confusion_matrix=np.load("./src/sft_cv/decision_system/assets/conf_mat.npy"),
    fake_targets=10,
    p_fake_targets=0.25,
    noise=0.0002,
    boundaries="./tools/simulators/assets/AirDrop/Maryland1GEarth.yaml",
)
