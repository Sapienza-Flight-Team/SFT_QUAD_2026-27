from .base import SensorsGenerator

generation = SensorsGenerator(
    gps_noise=4,
    barometer_noise=3,
    compass_noise=3,
    gyroscope_noise=1,
    data_path="./data/decision_system/Maryland",
    maps_path="./tools/simulators/assets/AirDrop/Maryland1GEarth.yaml",
    camera_path="./tools/simulators/assets/camera.yaml",
    time_step=0.1,
    lines_csv_path="./tools/simulators/assets/AirDrop/Maryland1GEarth.csv",
    ideal_speed=15.0,
    ideal_height=25.0,
    nominal_speed_noise=3.0,
    nominal_trajectory_brownianNoise=3.0,
    nominal_perspective_noise=0.3,
    nominal_height_brownianNoise=2.0,
    nominal_yaw_brownianNoise=1.0,
)
