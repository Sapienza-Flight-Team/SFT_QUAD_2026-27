import math
from pymavlink import mavutil

# Configure your connection string here
connection_string = "/dev/serial/by-id/usb-3D_Robotics_PX4_FMU_v5.x_0-if00"

print(f"Connecting to vehicle on: {connection_string}")
master = mavutil.mavlink_connection(connection_string)

print("Waiting for vehicle heartbeat...")
master.wait_heartbeat()
print("Heartbeat received!")

master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1,  # Arm the drone
    0,
    0,
    0,
    0,
    0,
    0,
)

print("Sending Activate (1.0) command to Actuator Set 2 (AUX 7)...")
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    187,
    0,  # confirmation
    math.nan,  # param1: Actuator Set 1 (Ignored)
    1.0,  # param2: Actuator Set 2 (AUX 7) -> 1.0 (Drop)
    math.nan,  # param3: Actuator Set 3
    math.nan,  # param4: Actuator Set 4
    math.nan,  # param5: Actuator Set 5
    math.nan,  # param6: Actuator Set 6
    0,  # param7: Index
)
print("Command sent. Payload dropped.")
