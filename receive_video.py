import subprocess

gst_command = [
    r"C:\Program Files\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe",
    "-v",
    "udpsrc", "port=5000", "caps=application/x-rtp,encoding-name=H264,payload=96",
    "!", "rtph264depay",
    "!", "avdec_h264",
    "!", "autovideosink"
]

subprocess.run(gst_command)