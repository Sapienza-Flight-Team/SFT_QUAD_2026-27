from core.options import Options

# With an external clock, camera notifications supply capture IDs and timing.
CONFIG = {
    "external_clock": Options((False, True), default=0),
    "generation_rate_hz": 60,
    "first_capture_id": 0,
    "latitude": 45.0,
    "longitude": 9.0,
    "coordinate_step": 0.0001,
    "altitude_msl": 150.0,
    "altitude_relative": 50.0,
    "waypoints": 10,
}
