import socket
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import threading
import time
from collections import deque
import ast  # Để chuyển đổi từ string JSON sang dict

# ======== UI SETTINGS ========
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOption('antialias', True)

# ======== QT APP ========
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Realtime Drone Telemetry")
win.resize(900, 700)
win.show()

# ======== PLOT PENS ========
pen_roll  = pg.mkPen(color=(0, 0, 255), width=2)
pen_pitch = pg.mkPen(color=(255, 0, 0), width=2)
pen_yaw   = pg.mkPen(color=(0, 150, 0), width=2)
pen_batt  = pg.mkPen(color=(255, 128, 0), width=2)

# ======== CREATE PLOTS ========
p_roll = win.addPlot(title="Roll (deg)")
curve_roll = p_roll.plot(pen=pen_roll)

win.nextRow()
p_pitch = win.addPlot(title="Pitch (deg)")
curve_pitch = p_pitch.plot(pen=pen_pitch)

win.nextRow()
p_yaw = win.addPlot(title="Yaw (deg)")
curve_yaw = p_yaw.plot(pen=pen_yaw)

win.nextRow()
p_batt = win.addPlot(title="Battery Voltage (V)")
curve_batt = p_batt.plot(pen=pen_batt)

# ======== DATA BUFFERS ========
MAX_POINTS = 600
t_list = deque(maxlen=MAX_POINTS)
roll_list = deque(maxlen=MAX_POINTS)
pitch_list = deque(maxlen=MAX_POINTS)
yaw_list = deque(maxlen=MAX_POINTS)
batt_list = deque(maxlen=MAX_POINTS)
start = time.time()

# ======== SOCKET CLIENT ========
CLIENT_IP = '100.105.38.109'  # Địa chỉ IP Tailscale của Raspberry Pi
CLIENT_PORT = 5000             # Cổng của server Raspberry Pi

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((CLIENT_IP, CLIENT_PORT))

# ======== RECEIVE DATA ========
buffer = ""

def receive_data():
    global buffer
    while True:
        chunk = client_socket.recv(1024).decode('utf-8')
        if not chunk:
            continue

        buffer += chunk

        # Xử lý từng dòng dữ liệu nhận được
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)

            if not line.strip():
                continue

            try:
                data_dict = ast.literal_eval(line)
                print(f"Received data: {data_dict}")  # Debugging
                t_data = time.time() - start

                # Cập nhật dữ liệu vào các list
                if data_dict['type'] == 'ATTITUDE':
                    roll_list.append(data_dict['roll'])
                    pitch_list.append(data_dict['pitch'])
                    yaw_list.append(data_dict['yaw'])
                    t_list.append(t_data)
                elif data_dict['type'] == 'SYS_STATUS':
                    batt_list.append(data_dict['battery'])
                    t_list.append(t_data)

            except Exception as e:
                print(f"Error: {e}")
        time.sleep(0.1)

# Start data receiving thread
threading.Thread(target=receive_data, daemon=True).start()

# ======== UPDATE UI ========
def update_ui():
    if not t_list:
        return

    t_data = list(t_list)
    curve_roll.setData(t_data, list(roll_list), fast=True)
    curve_pitch.setData(t_data, list(pitch_list), fast=True)
    
    # Normalize yaw so it starts from 0
    normalized_yaw = [yaw - yaw_list[0] for yaw in yaw_list]
    curve_yaw.setData(t_data, normalized_yaw, fast=True)

    if batt_list:
        curve_batt.setData(t_data[-len(batt_list):], list(batt_list), fast=True)

# ======== START TIMER ========
timer = QtCore.QTimer()
timer.timeout.connect(update_ui)
timer.start(16)  # 16 ms -> ~60 FPS

# ======== START APP ========
app.exec()
