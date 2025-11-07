# from flask import Flask, Response
# from picamera2 import Picamera2
# import cv2

# app = Flask(__name__)

# # Init camera
# picam2 = Picamera2()
# config = picam2.create_preview_configuration(main={"size": (640, 360)})
# picam2.configure(config)
# picam2.start()

# def generate():
#     while True:
#         frame = picam2.capture_array()
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         if not ret:
#             continue
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# @app.route('/video')
# def video_feed():
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/')
# def index():
#     return "<h1>Drone Camera Stream</h1><img src='/video' width='640'>"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, threaded=True)
from picamera2 import Picamera2, Preview
import cv2, time
from flask import Flask, Response

app = Flask(__name__)
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 360)})
picam2.configure(config)
picam2.start()

def generate():
    while True:
        request = picam2.capture_request()
        frame = request.make_array("main")
        request.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
