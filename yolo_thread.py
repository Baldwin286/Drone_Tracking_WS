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
    # no change, but print hint
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
        # store frame with lock
        with frame_lock:
            latest_frame = frame.copy()
        # tiny sleep to yield
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
            # Use model.predict so Ultralytics handles color/resize internally.
            # stream=False returns a Results list
            results = model.predict(source=frame_copy, conf=confidence, imgsz=imgsz, device=device, verbose=False)
            if results and len(results) > 0:
                with result_lock:
                    latest_result = results[0]  # Results object
            else:
                with result_lock:
                    latest_result = None
        except Exception as e:
            # print error but keep looping
            print("YOLO inference error:", e)
            with result_lock:
                latest_result = None
        # small sleep to limit inference rate
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

MAX_SPEED = 1.0
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

# def move_towards_person(cx, cy, frame_w, frame_h, area):
#     safe_distance_ratio = 0.15
#     frame_area = frame_w * frame_h
#     safe_area = frame_area * safe_distance_ratio

#     vy = (cx - frame_w / 2) / (frame_w / 2)   # horizontal -> rotate left/right
#     vz = -(cy - frame_h / 2) / (frame_h / 2)  # vertical -> up/down

#     # far -> move forward/near -> step back
#     dist_error = (safe_area - area) / safe_area
#     vx = np.clip(dist_error, -1, 1)

#     vx *= 0.5
#     vy *= 0.5
#     vz *= 0.5

#     send_ned_velocity(vx, vy, vz, duration=0.2)
def move_towards_person(cx, cy, frame_w, frame_h, area):
    """
    Duy trì khoảng cách an toàn dựa trên diện tích bbox.
    - SAFE_RATIO: tỉ lệ diện tích an toàn so với diện tích khung
    - HYSTERESIS: vùng chết (%) để tránh oscillation
    - Kp, min_speed, max_speed: tuning
    Mapping:
      vx > 0 -> move forward (approach)
      vx < 0 -> move backward (retreat)
      vy -> right/left (image x)
      vz -> up/down (image y) -- note: check sign mapping với NED nếu cần
    """
    # --- tham số (có thể tinh chỉnh) ---
    SAFE_RATIO = 0.15      # % diện tích frame mong muốn (thử 0.10..0.18)
    HYSTERESIS = 0.12      # 12% vùng chết
    KP_DIST = 0.9          # hệ số tỉ lệ cho tiến/lùi
    KP_LAT = 0.5           # hệ số tỉ lệ cho ngang
    KP_VERT = 0.45         # hệ số tỉ lệ cho dọc
    MIN_SPEED = 0.06       # vận tốc tối thiểu khi cần di chuyển
    # MAX_SPEED dùng global MAX_SPEED
    # -------------------------------

    frame_area = frame_w * frame_h
    safe_area = frame_area * SAFE_RATIO
    lower = safe_area * (1 - HYSTERESIS)
    upper = safe_area * (1 + HYSTERESIS)

    # --- lateral & vertical control (centre the person) ---
    vy = (cx - frame_w / 2) / (frame_w / 2)   # -1..1 -> left..right
    vz = -(cy - frame_h / 2) / (frame_h / 2)  # -1..1 -> down..up (you may invert if needed)
    vy_cmd = KP_LAT * vy
    vz_cmd = KP_VERT * vz

    # --- distance control (vx) with hysteresis ---
    if area < lower:
        # too far -> approach (vx > 0)
        err = (safe_area - area) / safe_area
        vx_cmd = KP_DIST * err
    elif area > upper:
        # too close -> retreat (vx < 0)
        err = (safe_area - area) / safe_area
        vx_cmd = KP_DIST * err
    else:
        # within safe band -> hold distance (no forward/back)
        vx_cmd = 0.0

    # ensure sign convention: when area > safe_area (too close), err negative => vx_cmd negative
    # but above err = (safe_area - area)/safe_area so will be negative when area>safe_area, so mapping OK

    # force a minimal speed so small commands have effect
    if 0 < abs(vx_cmd) < MIN_SPEED:
        vx_cmd = np.sign(vx_cmd) * MIN_SPEED

    # scale lateral/vertical so they don't exceed max
    vy_cmd = float(np.clip(vy_cmd, -MAX_SPEED, MAX_SPEED))
    vz_cmd = float(np.clip(vz_cmd, -MAX_SPEED, MAX_SPEED))
    vx_cmd = float(np.clip(vx_cmd, -MAX_SPEED, MAX_SPEED))

    print(f"[move] area={int(area)} safe={int(safe_area)} lower={int(lower)} upper={int(upper)} "
          f"vx={vx_cmd:.3f} vy={vy_cmd:.3f} vz={vz_cmd:.3f}")

    send_ned_velocity(vx_cmd, vy_cmd, vz_cmd, duration=0.2)

# ==================== main ==================== #
if __name__ == "__main__":
    try:
        # start threads BEFORE arming 
        t1 = Thread(target=capture_loop, daemon=True)
        t2 = Thread(target=detection_loop, daemon=True)
        t1.start()
        t2.start()

        # small wait to ensure camera ready
        time.sleep(0.5)

        if not arm_and_takeoff(5):
            print("Arming/takeoff failed")
            stop_threads = True
            t1.join(timeout=1.0)
            t2.join(timeout=1.0)
            vehicle.close()
            sys.exit(1)

        print("🎥 Starting main loop...")
        prev_time = time.time()

        while True:
            # get latest frame
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame = latest_frame.copy()

            # read latest result safely
            with result_lock:
                res = latest_result

            if res is not None and hasattr(res, "boxes") and len(res.boxes) > 0:
                try:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    scores = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy()
                except Exception as e:
                    # fallback if attributes differ
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

                    print(f"Person at ({cx},{cy}), area={int(area)}")
                    move_towards_person(cx, cy, frame.shape[1], frame.shape[0], area)
                    frame = draw_bounding_boxes(frame, person_boxes)
                else:
                    cv2.putText(frame, "No person (filtered)", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                cv2.putText(frame, "No detection", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("YOLO Drone Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        print("Landing...")
        vehicle.mode = VehicleMode("LAND")
        time.sleep(5)

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        stop_threads = True
        # give threads a moment to finish
        time.sleep(0.2)
        try:
            vehicle.mode = VehicleMode("LAND")
            time.sleep(2)
        except Exception:
            pass
        try:
            vehicle.close()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Closed and landed safely.")
