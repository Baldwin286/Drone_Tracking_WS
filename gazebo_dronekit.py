from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time, sys

CONN = 'udp:127.0.0.1:14550'

print("Connecting to vehicle on:", CONN)
vehicle = connect(CONN, wait_ready=True, heartbeat_timeout=30)

def arm_and_takeoff(aTargetAltitude):
    print("Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    t0 = time.time()
    while not vehicle.armed:
        if time.time() - t0 > 10:
            print("Timeout arming")
            return False
        print(" Waiting for arm...")
        time.sleep(0.5)

    print("Taking off to", aTargetAltitude, "m")
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(" Altitude:", round(alt,2))
        if alt >= aTargetAltitude * 0.95:
            print("Reached target")
            break
        time.sleep(0.5)
    return True

def send_ned_velocity(vx, vy, vz, duration=1.0):
    """
    Send velocity in m/s in the local NED frame
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,  # only velocity
        0,0,0,               # pos
        vx,vy,vz,            # velocity
        0,0,0,               # accel
        0,0)
    end = time.time() + duration
    while time.time() < end:
        vehicle.send_mavlink(msg)
        vehicle.flush()
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        if not arm_and_takeoff(5):
            print("Arming/takeoff failed")
            vehicle.close()
            sys.exit(1)

        print("Move forward for 3s")
        send_ned_velocity(1.0, 0, 0, duration=3.0)  # 1 m/s forward
        time.sleep(1)

        print("Hover")
        send_ned_velocity(0, 0, 0, duration=2.0)
        time.sleep(1)

        print("Return / Land")
        vehicle.mode = VehicleMode("LAND")
        time.sleep(5)

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        vehicle.close()
        print("Closed")
