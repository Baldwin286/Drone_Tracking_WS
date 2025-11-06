import cv2
import torch
import numpy as np
import time
import sys
from threading import Thread, Lock, Event
from collections import deque
from dronekit import connect, VehicleMode
from pymavlink import mavutil
from ultralytics import YOLO
import math
import logging

# =================== CONFIG ===================
CONN = '/dev/ttyAMA0'
BAUD = 921600
MODEL_PATH = '/home/phamthanhbien/yolo/my_model.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TAKEOFF_ALT = 1.0                
MAX_ALT = 3.0                    
MAX_SPEED = 0.4                
MAX_HORIZ_DIST = 20.0            
DETECTION_MIN_CONF = 0.5
DETECTION_TIMEOUT = 1.5         
COMMAND_RATE_HZ = 5
SMOOTHING_WINDOW = 5
ARM_TIMEOUT = 15
HEARTBEAT_TIMEOUT = 5

GST_PIPELINE = (
    "libcamera-vid "
    "--inline --codec=libav --width=640 --height=360 --framerate=30 --timeout=0 --nopreview "
    "-o - ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink"
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# =================== GLOBALS ===================
latest_frame = None
latest_result = None
frame_lock = Lock()
result_lock = Lock()
stop_event = Event()
cap = None
vehicle = None
home_location = None

cx_buf = deque(maxlen=SMOOTHING_WINDOW)
cy_buf = deque(maxlen=SMOOTHING_WINDOW)
area_buf = deque(maxlen=SMOOTHING_WINDOW)

last_detection_time = 0.0
last_command_time = 0.0
send_fail_count = 0

# =================== UTILS ===================
def get_distance_meters(loc1, loc2):
    R = 6371000.0
    lat1, lon1 = float(loc1.lat), float(loc1.lon)
    lat2, lon2 = float(loc2.lat), float(loc2.lon)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def safe_send_mavlink(msg):
    global send_fail_count
    try:
        vehicle.send_mavlink(msg)
        vehicle.flush()
        send_fail_count = 0
        return True
    except Exception as e:
        send_fail_count += 1
        logging.warning(f"MAVLink send error #{send_fail_count}: {e}")
        return False

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
        safe_send_mavlink(msg)
        time.sleep(0.05)
    return True

# =================== CAMERA THREAD ===================
def capture_loop():
    global latest_frame, cap
    cap_local = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap_local.isOpened():
        logging.error("Failed to open Pi Camera V3 stream!")
        stop_event.set()
        return
    logging.info("Pi Camera streaming started.")
    cap = cap_local
    while not stop_event.is_set():
        ret, frame = cap_local.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        time.sleep(0.01)
    cap_local.release()
    logging.info("Camera stopped.")

# =================== YOLO THREAD ===================
def detection_loop(conf=DETECTION_MIN_CONF, imgsz=640):
    global latest_result, last_detection_time
    while not stop_event.is_set():
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue
        try:
            results = model.predict(source=frame, conf=conf, imgsz=imgsz, device=DEVICE, verbose=False)
            if results and len(results) > 0:
                with result_lock:
                    latest_result = results[0]
                    last_detection_time = time.time()
            else:
                with result_lock:
                    latest_result = None
        except Exception as e:
            logging.warning(f"YOLO error: {e}")
            latest_result = None
        time.sleep(0.01)

# =================== MOVEMENT ===================
def move_towards_person(cx, cy, frame_w, frame_h, area):
    """Giữ độ cao cố định (vz=0). Drone chỉ di chuyển ngang & tới-lùi."""
    SAFE_RATIO = 0.50
    HYSTERESIS = 0.12
    KP_DIST = 0.6
    KP_LAT = 0.5
    MIN_SPEED = 0.06

    frame_area = frame_w * frame_h
    safe_area = frame_area * SAFE_RATIO
    lower, upper = safe_area * (1 - HYSTERESIS), safe_area * (1 + HYSTERESIS)

    vy = (cx - frame_w / 2) / (frame_w / 2)
    vy_cmd = KP_LAT * vy

    if area < lower:
        vx_cmd = KP_DIST * ((safe_area - area) / safe_area)
    elif area > upper:
        vx_cmd = -KP_DIST * ((area - safe_area) / safe_area)
    else:
        vx_cmd = 0.0

    vz_cmd = 0.0  

    if 0 < abs(vx_cmd) < MIN_SPEED:
        vx_cmd = np.sign(vx_cmd) * MIN_SPEED
    if 0 < abs(vy_cmd) < MIN_SPEED:
        vy_cmd = np.sign(vy_cmd) * MIN_SPEED

    vx_cmd, vy_cmd, vz_cmd = [float(np.clip(v, -MAX_SPEED, MAX_SPEED)) for v in (vx_cmd, vy_cmd, vz_cmd)]
    logging.debug(f"[move-flat] vx={vx_cmd:.2f}, vy={vy_cmd:.2f}")

    try:
        alt = vehicle.location.global_relative_frame.alt
        if alt > MAX_ALT:
            logging.warning(f"Altitude {alt:.1f} > MAX_ALT. Hovering.")
            return send_ned_velocity(0, 0, 0, 0.2)
    except Exception:
        pass

    if home_location:
        cur_loc = vehicle.location.global_frame
        dist = get_distance_meters(cur_loc, home_location)
        if dist > MAX_HORIZ_DIST:
            logging.warning(f"Exceeded geofence ({dist:.1f}m). RTL.")
            vehicle.mode = VehicleMode("RTL")
            return False

    return send_ned_velocity(vx_cmd, vy_cmd, vz_cmd, 1.0 / COMMAND_RATE_HZ)

# =================== TAKEOFF ===================
def arm_and_takeoff(target_alt):
    logging.info("Arming motors...")
    t0 = time.time()
    while not vehicle.is_armable and time.time() - t0 < ARM_TIMEOUT:
        logging.info(" Waiting for vehicle to initialize...")
        time.sleep(1)
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed and time.time() - t0 < ARM_TIMEOUT:
        logging.info(" Waiting for arming...")
        time.sleep(1)
    logging.info(f"Taking off to {target_alt}m...")
    vehicle.simple_takeoff(target_alt)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        logging.info(f" Altitude: {alt:.2f}m")
        if alt >= target_alt * 0.95:
            logging.info("Reached target altitude")
            break
        time.sleep(0.5)
    return True

# =================== MAIN ===================
if __name__ == "__main__":
    try:
        logging.info(f"Connecting to vehicle at {CONN}...")
        vehicle = connect(CONN, baud=BAUD, wait_ready=True, heartbeat_timeout=60)
        logging.info("Connected to FCU.")
        home_location = vehicle.location.global_frame
        logging.info(f"Home: {home_location.lat:.6f}, {home_location.lon:.6f}")

        model = YOLO(MODEL_PATH).to(DEVICE)
        logging.info(f"YOLO model loaded on {DEVICE}")

        # Threads
        t1 = Thread(target=capture_loop, daemon=True)
        t2 = Thread(target=detection_loop, daemon=True)
        t1.start(); t2.start()
        time.sleep(0.5)

        if not arm_and_takeoff(TAKEOFF_ALT):
            logging.error("Takeoff failed.")
            sys.exit(1)

        logging.info("Starting main loop at 1m altitude...")

        prev_time = time.time()

        while not stop_event.is_set():
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("User quit -> LAND.")
                vehicle.mode = VehicleMode("LAND")
                break
            if key == ord('s'):
                logging.warning("EMERGENCY STOP -> RTL.")
                vehicle.mode = VehicleMode("RTL")
                stop_event.set()
                break

            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame = latest_frame.copy()

            with result_lock:
                res = latest_result

            now = time.time()
            fps = 1.0 / (now - prev_time)
            prev_time = now

            if res is not None and hasattr(res, "boxes") and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()

                persons = [(b, s) for b, s, c in zip(boxes, scores, classes) if int(c) == 0 and s > DETECTION_MIN_CONF]
                if persons:
                    box, score = sorted(persons, key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]), reverse=True)[0]
                    x1, y1, x2, y2 = map(int, box[:4])
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    area = (x2-x1)*(y2-y1)

                    cx_buf.append(cx); cy_buf.append(cy); area_buf.append(area)
                    scx, scy, sarea = np.mean(cx_buf), np.mean(cy_buf), np.mean(area_buf)

                    logging.info(f"Person ({scx:.0f},{scy:.0f}) area={sarea:.0f}")
                    move_towards_person(scx, scy, frame.shape[1], frame.shape[0], sarea)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                else:
                    cv2.putText(frame, "No person", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                cv2.putText(frame, "No detection", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            if time.time() - last_detection_time > DETECTION_TIMEOUT:
                logging.info("No detection -> hover.")
                send_ned_velocity(0, 0, 0, 0.3)

            alt = vehicle.location.global_relative_frame.alt
            cv2.putText(frame, f"Alt: {alt:.1f}m FPS:{fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            cv2.imshow("YOLO Drone Tracking (1m fixed)", frame)

        logging.info("Landing...")
        vehicle.mode = VehicleMode("LAND")
        time.sleep(3)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        stop_event.set()
        try:
            vehicle.mode = VehicleMode("LAND")
            vehicle.close()
        except Exception:
            pass
        if cap: cap.release()
        cv2.destroyAllWindows()
        logging.info("Exited safely.")
