import subprocess
import cv2

gst_command = [
    r"C:\Program Files\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe",
    "-v",
    "udpsrc", "port=5000", "caps=application/x-rtp,encoding-name=H264,payload=96",
    "!", "rtph264depay",
    "!", "avdec_h264",
    "!", "autovideosink"
]

subprocess.run(gst_command)

gst_str = (
    "udpsrc port=5600 ! "
    "application/x-rtp, encoding-name=H264,payload=96 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! "
    "videoconvert ! appsink"
)

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open video stream")
    exit(1)
else:
    print("Video stream opened")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Video", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

