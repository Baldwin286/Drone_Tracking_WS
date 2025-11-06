# Author: Pham Thanh Bien
# Date: 2025-09-25
# Description: A simple tracking drone using YOLO model in Gazebo simulation
# Run terminal: python thead_tracking_gazebo_sim.py with gz and ardupilot SITL

import cv2
import torch
import numpy as np
import time
import sys
from threading import Thread, Lock
from dronekit import connect, VehicleMode
from pymavlink import mavutil
from ultralytics import YOLO

# ==================== connect vehicle gazebo ==================== #
CONN = 'udp:127.0.0.1:14550'
print("Connecting to vehicle on:", CONN)
vehicle = connect(CONN, wait_ready=True, heartbeat_timeout=30)
print("Connected!")

# ==================== MODEL YOLO ==================== #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

MODEL_PATH = '/home/phamthanhbien/yolo/my_model.pt'
if not torch.cuda.is_available():
    print("Note: CUDA not available -> model will run on CPU (slower).")

model = YOLO(MODEL_PATH).to(device)

# ==================== global variables ==================== #
latest_frame = None
latest_result = None
stop_threads = False
frame_lock = Lock()
result_lock = Lock()
cap = None

# ==================== read camera thread ==================== #
def capture_loop():
    global latest_frame, cap, stop_threads
    cap_local = cv2.VideoCapture(0)
    cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap = cap_local
    while not stop_threads:
        ret, frame = cap_local.read()
        if not ret:
            time.sleep(0.01)
            continue
        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(0.005)
    cap_local.release()

# ==================== yolo detection thread ==================== #
def detection_loop(confidence=0.5, imgsz=640):
    global latest_result, stop_threads
    while not stop_threads:
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
            else:
                with result_lock:
                    latest_result = None
        except Exception as e:
            print("YOLO inference error:", e)
            with result_lock:
                latest_result = None
        time.sleep(0.01)

# ==================== support ==================== #
def draw_bounding_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

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

MAX_SPEED = 1
def send_ned_velocity(vx, vy, vz, duration=0.2):
    vx = float(np.clip(vx, -MAX_SPEED, MAX_SPEED))
    vy = float(np.clip(vy, -MAX_SPEED, MAX_SPEED))
    vz = float(np.clip(vz, -MAX_SPEED, MAX_SPEED))
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0)
    end = time.time() + duration
    while time.time() < end:
        vehicle.send_mavlink(msg)
        vehicle.flush()
        time.sleep(0.05)

# ==================== move towards person (fixed altitude) ==================== #
FIXED_ALTITUDE = 1.2  # meters
def move_towards_person(cx, cy, frame_w, frame_h, area):
    SAFE_RATIO = 0.50
    HYSTERESIS = 0.12
    KP_DIST = 0.6
    KP_LAT = 0.5
    MIN_SPEED = 0.06

    frame_area = frame_w * frame_h
    safe_area = frame_area * SAFE_RATIO
    lower = safe_area * (1 - HYSTERESIS)
    upper = safe_area * (1 + HYSTERESIS)

    # lateral offset [-1..1]
    vy = (cx - frame_w / 2) / (frame_w / 2)
    vy_cmd = KP_LAT * vy

    # forward/backward
    if area < lower:
        err = (safe_area - area) / safe_area
        vx_cmd = KP_DIST * err
    elif area > upper:
        err = (area - safe_area) / safe_area
        vx_cmd = -KP_DIST * err
    else:
        vx_cmd = 0.0

    # minimum speed
    if 0 < abs(vx_cmd) < MIN_SPEED:
        vx_cmd = np.sign(vx_cmd) * MIN_SPEED

    vx_cmd = float(np.clip(vx_cmd, -MAX_SPEED, MAX_SPEED))
    vy_cmd = float(np.clip(vy_cmd, -MAX_SPEED, MAX_SPEED))
    vz_cmd = 0.0  # keep altitude

    print(f"[move] area={int(area)} safe={int(safe_area)} vx={vx_cmd:.3f} vy={vy_cmd:.3f}")

    send_ned_velocity(vx_cmd, vy_cmd, vz_cmd, duration=0.2)

# ==================== main ==================== #
if __name__ == "__main__":
    try:
        t1 = Thread(target=capture_loop, daemon=True)
        t2 = Thread(target=detection_loop, daemon=True)
        t1.start()
        t2.start()

        time.sleep(0.5)

        if not arm_and_takeoff(FIXED_ALTITUDE):
            print("Arming/takeoff failed")
            stop_threads = True
            t1.join(timeout=1.0)
            t2.join(timeout=1.0)
            vehicle.close()
            sys.exit(1)

        print("Starting main loop...")
        prev_time = time.time()
        last_seen_time = time.time()

        while True:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame = latest_frame.copy()

            with result_lock:
                res = latest_result

            if res is not None and hasattr(res, "boxes") and len(res.boxes) > 0:
                try:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    scores = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy()
                except:
                    boxes = np.array([])
                    scores = np.array([])
                    classes = np.array([])

                person_boxes = [
                    box for box, score, cls in zip(boxes, scores, classes)
                    if int(cls) == 0 and score > 0.5
                ]

                if len(person_boxes) > 0:
                    x1, y1, x2, y2 = person_boxes[0][:4]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    area = (x2 - x1) * (y2 - y1)

                    last_seen_time = time.time()
                    move_towards_person(cx, cy, frame.shape[1], frame.shape[0], area)
                    frame = draw_bounding_boxes(frame, person_boxes)
                else:
                    cv2.putText(frame, "No person", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                cv2.putText(frame, "No detection", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # auto-land if not seen person for >5s
            if time.time() - last_seen_time > 5:
                print("Person not seen >5s, landing...")
                vehicle.mode = VehicleMode("LAND")
                break

            now = time.time()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            cv2.imshow("YOLO Drone Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Landing by user request...")
                vehicle.mode = VehicleMode("LAND")
                break

        time.sleep(5)

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        stop_threads = True
        time.sleep(0.2)
        try:
            vehicle.mode = VehicleMode("LAND")
            time.sleep(2)
        except Exception as e:
            print(f"Error during LAND: {e}")
        try:
            vehicle.close()
            print("Vehicle connection closed.")
        except Exception as e:
            print(f"Error closing vehicle: {e}")
        try:
            if cap is not None:
                cap.release()
        except Exception as e:
            print(f"Error releasing camera: {e}")
        cv2.destroyAllWindows()
        print("Closed and landed safely.")
