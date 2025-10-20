import cv2
import torch
import numpy as np
import time
import sys
from threading import Thread
from dronekit import connect, VehicleMode
from pymavlink import mavutil
from ultralytics import YOLO

# ==================== connect vehicle gazebo ==================== #
CONN = 'udp:127.0.0.1:14550'
print("🔗 Connecting to vehicle on:", CONN)
vehicle = connect(CONN, wait_ready=True, heartbeat_timeout=30)
print("Connected!")

# ==================== MODEL YOLO ==================== #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

model = YOLO('/home/phamthanhbien/yolo/my_model.pt').to(device)

# ==================== global variables ==================== #
latest_frame = None
latest_result = None
stop_threads = False

# ==================== read camera thread ==================== #
def capture_loop():
    global latest_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    while not stop_threads:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame
        time.sleep(0.01)  
    cap.release()

# ==================== yolo detection threads ==================== #
def detection_loop():
    global latest_result
    while not stop_threads:
        if latest_frame is not None:
            img = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 360))
            results = model(img)
            latest_result = results[0]
        time.sleep(0.05)  

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

def send_ned_velocity(vx, vy, vz, duration=0.2):
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

def move_towards_person(cx, cy, frame_w, frame_h, area):
    safe_distance_ratio = 0.15
    frame_area = frame_w * frame_h

    vx = -(cy - frame_h / 2) / (frame_h / 2)
    vy = (cx - frame_w / 2) / (frame_w / 2)
    vz = 0

    vx, vy = np.clip(vx, -1, 1), np.clip(vy, -1, 1)

    if area < frame_area * safe_distance_ratio:
        send_ned_velocity(vx * 0.5, vy * 0.5, vz, duration=0.2)
    else:
        send_ned_velocity(-vx * 0.5, -vy * 0.5, vz, duration=0.2)

# ==================== main ==================== #
if __name__ == "__main__":
    try:
        if not arm_and_takeoff(5):
            print("Arming/takeoff failed")
            vehicle.close()
            sys.exit(1)

        # start threads
        Thread(target=capture_loop, daemon=True).start()
        Thread(target=detection_loop, daemon=True).start()

        print("🎥 Starting main loop...")   
        prev_time = time.time()

        while True:
            if latest_frame is None:
                continue

            frame = latest_frame.copy()

            if latest_result is not None:
                boxes = latest_result.boxes.xyxy.cpu().numpy()
                scores = latest_result.boxes.conf.cpu().numpy()
                classes = latest_result.boxes.cls.cpu().numpy()

                person_boxes = [
                    box for box, score, cls in zip(boxes, scores, classes)
                    if int(cls) == 0 and score > 0.5
                ]

                if len(person_boxes) > 0:
                    x1, y1, x2, y2 = person_boxes[0][:4]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    area = (x2 - x1) * (y2 - y1)

                    print(f"🎯 Person at ({cx},{cy}), area={int(area)}")
                    move_towards_person(cx, cy, frame.shape[1], frame.shape[0], area)
                    frame = draw_bounding_boxes(frame, person_boxes)

            # FPS
            now = time.time()
            fps = 1 / (now - prev_time)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("YOLO Drone Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Landing...")
        vehicle.mode = VehicleMode("LAND")
        time.sleep(5)

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        stop_threads = True
        vehicle.mode = VehicleMode("LAND")
        time.sleep(5)
        vehicle.close()
        cv2.destroyAllWindows()
        print("Closed and landed safely.")
