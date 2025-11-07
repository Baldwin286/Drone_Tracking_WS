# from picamera2 import Picamera2
# import cv2
# import time

# def main():
#     print("Initializing Pi Camera V3...")
#     picam2 = Picamera2()

#     # Create preview configuration (adjust size if needed)
#     config = picam2.create_preview_configuration(
#         main={"size": (640, 360), "format": "RGB888"}
#     )
#     picam2.configure(config)
#     picam2.start()

#     print("Pi Camera started successfully!")
#     print("Press 'q' to quit.")

#     prev_time = time.time()
#     fps = 0.0

#     while True:
#         frame = picam2.capture_array()
#         now = time.time()
#         fps = 1.0 / (now - prev_time)
#         prev_time = now

#         # Show FPS
#         cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#         cv2.imshow("Pi Camera V3 Test", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     print("Stopping camera...")
#     picam2.stop()
#     cv2.destroyAllWindows()
#     print("Camera test ended.")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Stream Pi Camera (Picamera2) to browser over Tailscale network
Author: Pham Thanh Bien
"""

from flask import Flask, Response
from picamera2 import Picamera2
import cv2

app = Flask(__name__)

# Init camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 360)})
picam2.configure(config)
picam2.start()

def generate():
    while True:
        frame = picam2.capture_array()
        # Nếu bạn có YOLO tracking, bạn có thể vẽ bounding box ở đây
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(frame, "target", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Drone Camera Stream</h1><img src='/video' width='640'>"

if __name__ == '__main__':
    # Lắng nghe trên tất cả IP, kể cả IP Tailscale
    app.run(host='0.0.0.0', port=5000, threaded=True)
