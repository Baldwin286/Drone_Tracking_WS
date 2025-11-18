import cv2
import torch
import numpy as np
import time
import sys
from threading import Thread, Lock
from dronekit import connect, VehicleMode
from pymavlink import mavutil
from ultralytics import YOLO
from picamera2 import Picamera2
from flask import Flask, Response

# ==================== connect vehicle via UART ==================== #
CONN = '/dev/ttyAMA0'
BAUD = 921600

print(f"Connecting to vehicle on: {CONN} at {BAUD} baud")
vehicle = connect(CONN, baud=BAUD, wait_ready=False)
print("Connected to vehicle (wait_ready=False)")
vehicle.parameters['ARMING_CHECK'] = 0  # disable arming checks

# ==================== MODEL YOLO ==================== #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
MODEL_PATH = '/home/bien/Drone_Tracking_WS/model/my_model.pt'
model = YOLO(MODEL_PATH).to(device)

# ==================== global variables ==================== #
latest_frame = None
latest_result = None
stop_threads = False
frame_lock = Lock()
result_lock = Lock()

# ==================== Flask streaming ==================== #
app = Flask(__name__)

def generate_stream():
    global latest_frame
    while not stop_threads:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)

@app.route('/video')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>YOLO Drone Tracking Stream</h1><img src='/video' width='640'>"

def flask_stream_thread():
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)

# ==================== camera capture ==================== #
def capture_loop():
    global latest_frame, stop_threads
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 360), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        print("Pi Camera started successfully!")
        while not stop_threads:
            frame = picam2.capture_array()
            with frame_lock:
                latest_frame = frame.copy()
            time.sleep(0.01)
        picam2.stop()
    except Exception as e:
        print(f"Camera init failed: {e}")
        stop_threads = True

# ==================== YOLO detection ==================== #
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
            results = model.predict(source=frame_copy, conf=confidence, imgsz=imgsz,
                                    device=device, verbose=False)
            with result_lock:
                latest_result = results[0] if results else None
        except Exception as e:
            print("YOLO inference error:", e)
            with result_lock:
                latest_result = None
        time.sleep(0.01)

# ==================== helper functions ==================== #
def draw_bounding_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def arm_motors():
    print("Arming motors...")
    vehicle.mode = VehicleMode("STABILIZE")
    vehicle.armed = True
    t0 = time.time()
    while not vehicle.armed:
        if time.time() - t0 > 10:
            print("Timeout arming")
            return False
        print(" Waiting for arm...")
        time.sleep(0.5)
    print("Motors armed.")
    return True

def keep_motor_alive(thrust=0.1):
    """
    Send attitude/thrust messages repeatedly to keep motors spinning.
    thrust: 0.0~1.0
    """
    msg = vehicle.message_factory.set_attitude_target_encode(
        0,                  # time_boot_ms
        0, 0,               # target system, target component
        0b00000000,         # type_mask (all rates disabled)
        [0, 0, 0, 1],       # quaternion (no rotation)
        0, 0, 0,            # body roll, pitch, yaw rate
        thrust               # thrust
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

# ==================== main ==================== #
if __name__ == "__main__":
    try:
        # start threads
        Thread(target=capture_loop, daemon=True).start()
        Thread(target=detection_loop, daemon=True).start()
        Thread(target=flask_stream_thread, daemon=True).start()
        time.sleep(0.5)  # camera warmup

        if not arm_motors():
            print("Arming failed, exiting...")
            stop_threads = True
            sys.exit(1)

        print("Starting manual tracking loop...")
        prev_time = time.time()
        while True:
            keep_motor_alive(0.1)  # giữ motor quay với lực nhỏ

            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame = latest_frame.copy()

            with result_lock:
                res = latest_result

            person_boxes = []
            if res is not None and hasattr(res, "boxes") and len(res.boxes) > 0:
                try:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    scores = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy()
                    person_boxes = [box for box, score, cls in zip(boxes, scores, classes)
                                    if int(cls) == 0 and score > 0.5]
                except Exception:
                    pass

            if len(person_boxes) > 0:
                x1, y1, x2, y2 = person_boxes[0][:4]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                area = (x2 - x1) * (y2 - y1)
                print(f"Person at ({cx},{cy}), area={int(area)})")
                frame = draw_bounding_boxes(frame, person_boxes)
            else:
                cv2.putText(frame, "No person (filtered)", (10,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-5)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        stop_threads = True
        time.sleep(0.2)
        print("Disarming motors...")
        try:
            vehicle.armed = False
        except Exception as e:
            print(f"Error disarming: {e}")
        vehicle.close()
        print("Closed safely.")
