import cv2
import torch
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
from flask import Flask, Response
from threading import Thread, Lock
import time

# ==================== MODEL YOLO ==================== #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

MODEL_PATH = '/home/bien/Drone_Tracking_WS/model/my_model.pt'
model = YOLO(MODEL_PATH).to(device)

# ==================== Global variables ==================== #
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
    return "<h1>YOLO Detect Test Stream</h1><img src='/video' width='640'>"

def flask_thread():
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)

# ==================== Pi Camera capture ==================== #
def capture_loop():
    global latest_frame, stop_threads
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 360), "format": "RGB888"}
        )
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
        print("Camera init failed:", e)
        stop_threads = True

# ==================== YOLO detection ==================== #
def detection_loop(confidence=0.5, imgsz=640):
    global latest_frame, latest_result, stop_threads
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

def draw_bboxes(frame, res):
    if res is None or not hasattr(res, "boxes"):
        return frame
    try:
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()
    except Exception:
        return frame
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{int(cls)} {score:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

# ==================== Main ==================== #
if __name__ == "__main__":
    t1 = Thread(target=capture_loop, daemon=True)
    t2 = Thread(target=detection_loop, daemon=True)
    t3 = Thread(target=flask_thread, daemon=True)

    t1.start()
    t2.start()
    t3.start()

    try:
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()
            with result_lock:
                res = latest_result
            frame = draw_bboxes(frame, res)
            # Optional local preview
            cv2.imshow("YOLO Detect Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_threads = True
        time.sleep(0.2)
        cv2.destroyAllWindows()
        print("Test detect stopped.")
