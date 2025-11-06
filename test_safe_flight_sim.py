import cv2
import torch
import numpy as np
import time
import sys
from threading import Thread, Lock, Event
from dronekit import connect, VehicleMode
from pymavlink import mavutil
from ultralytics import YOLO
import logging

# ==================== CONFIG ==================== #
CONN = 'udp:127.0.0.1:14550'
MODEL_PATH = '/home/phamthanhbien/yolo/my_model.pt'
TAKEOFF_ALT = 1.0        
MAX_SPEED = 0.5             # m/s
DETECTION_TIMEOUT = 8.0     
SAFE_RATIO = 0.45
HYSTERESIS = 0.15
ARM_TIMEOUT = 15
COMMAND_RATE_HZ = 5

# ==================== INIT ==================== #
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

print("Connecting to vehicle on:", CONN)
vehicle = connect(CONN, wait_ready=True, heartbeat_timeout=60)
print("Connected to SITL Gazebo FCU!")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
model = YOLO(MODEL_PATH).to(device)

# ==================== GLOBALS ==================== #
latest_frame = None
latest_result = None
frame_lock = Lock()
result_lock = Lock()
stop_event = Event()
cap = None
last_detection_time = time.time()

# ==================== DRONE CONTROL ==================== #
def send_ned_velocity(vx, vy, vz, duration=0.2):
    vx, vy, vz = [float(np.clip(v, -MAX_SPEED, MAX_SPEED)) for v in (vx, vy, vz)]
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )
    end = time.time() + duration
    while time.time() < end and not stop_event.is_set():
        try:
            vehicle.send_mavlink(msg)
            vehicle.flush()
        except:
            pass
        time.sleep(0.05)

def arm_and_takeoff(aTargetAltitude):
    print("Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    t0 = time.time()
    while not vehicle.armed:
        if time.time() - t0 > ARM_TIMEOUT:
            print("Timeout arming")
            return False
        print(" Waiting for arm...")
        time.sleep(0.5)
    print(f"Taking off to {aTargetAltitude} m")
    vehicle.simple_takeoff(aTargetAltitude)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(" Altitude:", round(alt, 2))
        if alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(0.5)
    return True

# ==================== CAMERA THREAD ==================== #
def capture_loop():
    global latest_frame, cap
    cap_local = cv2.VideoCapture(0)
    cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap = cap_local
    print("Camera stream started.")
    while not stop_event.is_set():
        ret, frame = cap_local.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            time.sleep(0.01)
    cap_local.release()
    print("Camera stopped.")

# ==================== YOLO THREAD ==================== #
def detection_loop(confidence=0.5, imgsz=640):
    global latest_result, last_detection_time
    while not stop_event.is_set():
        frame_copy = None
        with frame_lock:
            if latest_frame is not None:
                frame_copy = latest_frame.copy()
        if frame_copy is None:
            time.sleep(0.01)
            continue
        try:
            results = model.predict(source=frame_copy, conf=confidence, imgsz=imgsz, device=device, verbose=False)
            if results and len(results) > 0:
                with result_lock:
                    latest_result = results[0]
                    last_detection_time = time.time()
            else:
                with result_lock:
                    latest_result = None
        except Exception as e:
            print("YOLO inference error:", e)
            with result_lock:
                latest_result = None
        time.sleep(0.01)

# ==================== MOVE LOGIC ==================== #
def move_towards_person(cx, cy, frame_w, frame_h, area):
    frame_area = frame_w * frame_h
    safe_area = frame_area * SAFE_RATIO
    lower = safe_area * (1 - HYSTERESIS)
    upper = safe_area * (1 + HYSTERESIS)

    vy = (cx - frame_w / 2) / (frame_w / 2)
    vy_cmd = 0.5 * vy

    if area < lower:
        vx_cmd = 0.6 * ((safe_area - area) / safe_area)
    elif area > upper:
        vx_cmd = -0.6 * ((area - safe_area) / safe_area)
    else:
        vx_cmd = 0.0

    vz_cmd = 0.0
    vx_cmd, vy_cmd, vz_cmd = [float(np.clip(v, -MAX_SPEED, MAX_SPEED)) for v in (vx_cmd, vy_cmd, vz_cmd)]
    print(f"[move] vx={vx_cmd:.2f} vy={vy_cmd:.2f} area={int(area)} safe={int(safe_area)}")
    send_ned_velocity(vx_cmd, vy_cmd, vz_cmd, duration=1.0 / COMMAND_RATE_HZ)

# ==================== MAIN ==================== #
if __name__ == "__main__":
    try:
        t1 = Thread(target=capture_loop, daemon=True)
        t2 = Thread(target=detection_loop, daemon=True)
        t1.start()
        t2.start()
        time.sleep(0.5)

        if not arm_and_takeoff(TAKEOFF_ALT):
            print("Takeoff failed.")
            sys.exit(1)

        print("Starting tracking loop...")
        prev_time = time.time()

        while not stop_event.is_set():
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame = latest_frame.copy()

            with result_lock:
                res = latest_result

            now = time.time()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
            prev_time = now

            if res is not None and hasattr(res, "boxes") and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()

                person_boxes = [b for b, s, c in zip(boxes, scores, classes) if int(c) == 0 and s > 0.5]
                if len(person_boxes) > 0:
                    x1, y1, x2, y2 = person_boxes[0][:4].astype(int)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    area = (x2 - x1) * (y2 - y1)

                    move_towards_person(cx, cy, frame.shape[1], frame.shape[0], area)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, "Tracking", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                cv2.putText(frame, "No detection", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            if time.time() - last_detection_time > DETECTION_TIMEOUT:
                print(f"No detection for {DETECTION_TIMEOUT:.0f}s → AUTO LAND")
                vehicle.mode = VehicleMode("LAND")
                break

            cv2.putText(frame, f"FPS:{fps:.1f} Alt:{vehicle.location.global_relative_frame.alt:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.imshow("YOLO Gazebo Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User quit → LAND")
                vehicle.mode = VehicleMode("LAND")
                break
            if key == ord('s'):
                print("RTL triggered!")
                vehicle.mode = VehicleMode("RTL")
                break

        print("Landing...")
        vehicle.mode = VehicleMode("LAND")
        time.sleep(3)

    except KeyboardInterrupt:
        print("Keyboard interrupt → LAND")
        vehicle.mode = VehicleMode("LAND")

    finally:
        stop_event.set()
        try:
            vehicle.mode = VehicleMode("LAND")
            vehicle.close()
        except:
            pass
        if cap: cap.release()
        cv2.destroyAllWindows()
        print("Closed and landed safely.")
