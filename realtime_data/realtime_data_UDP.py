# from pymavlink import mavutil
# import matplotlib.pyplot as plt
# import time

# # ================== CONNECT MAVLINK =====================
# mav = mavutil.mavlink_connection('udp:0.0.0.0:14550')
# print("Waiting for heartbeat...")
# mav.wait_heartbeat()
# print("Connected!")

# # ================== INIT GRAPH =========================
# plt.ion()
# fig, axs = plt.subplots(4, 1, figsize=(8, 10))

# # Initialize empty lines
# line_roll,   = axs[0].plot([], [])
# line_pitch,  = axs[1].plot([], [])
# line_yaw,    = axs[2].plot([], [])
# line_batt,   = axs[3].plot([], [])

# axs[0].set_title("Roll")
# axs[1].set_title("Pitch")
# axs[2].set_title("Yaw")
# axs[3].set_title("Battery Voltage (V)")

# roll_list = []
# pitch_list = []
# yaw_list = []
# battery_list = []
# t_list = []

# start_time = time.time()

# try:
#     while True:
#         msg = mav.recv_match(type=['ATTITUDE', 'SYS_STATUS'], blocking=True)
#         now = time.time() - start_time

#         if msg.get_type() == "ATTITUDE":
#             roll_list.append(msg.roll)
#             pitch_list.append(msg.pitch)
#             yaw_list.append(msg.yaw)
#             t_list.append(now)

#         if msg.get_type() == "SYS_STATUS":
#             voltage = msg.voltage_battery / 1000.0
#             battery_list.append(voltage)

#         # Update data without clearing
#         line_roll.set_xdata(t_list)
#         line_roll.set_ydata(roll_list)

#         line_pitch.set_xdata(t_list)
#         line_pitch.set_ydata(pitch_list)

#         line_yaw.set_xdata(t_list)
#         line_yaw.set_ydata(yaw_list)

#         if len(battery_list) > 0:
#             line_batt.set_xdata(t_list[:len(battery_list)])
#             line_batt.set_ydata(battery_list)

#         for ax in axs:
#             ax.relim()
#             ax.autoscale_view()

#         plt.pause(0.01)

# except KeyboardInterrupt:
#     print("Stopped by user.")
#     plt.close()
from pymavlink import mavutil
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import threading
import time

# ======== UI ĐẸP NHƯ MATPLOTLIB ========
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOption('antialias', True)

# ======== CONNECT MAVLINK ========
mav = mavutil.mavlink_connection('udp:0.0.0.0:14550')
print("Waiting for heartbeat...")
mav.wait_heartbeat()
print("Connected!")

# ======== QT APP ========
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Realtime Drone Telemetry")
win.resize(900, 700)
win.show()

# ======== Plot Pens ========
pen_roll  = pg.mkPen(color=(0, 0, 255), width=2)
pen_pitch = pg.mkPen(color=(255, 0, 0), width=2)
pen_yaw   = pg.mkPen(color=(0, 150, 0), width=2)
pen_batt  = pg.mkPen(color=(255, 128, 0), width=2)

# Plots
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
t_list = []
roll_list = []
pitch_list = []
yaw_list = []
batt_list = []

MAX_POINTS = 600
start = time.time()

# ======== THREAD: READ MAVLINK (KHÔNG BỊ MẤT GÓI) ========
def mav_reader():
    while True:
        msg = mav.recv_match(type=["ATTITUDE", "SYS_STATUS"], blocking=True)

        now = time.time() - start

        if msg.get_type() == "ATTITUDE":
            t_list.append(now)
            roll_list.append(msg.roll * 57.2958)
            pitch_list.append(msg.pitch * 57.2958)
            yaw_list.append(msg.yaw * 57.2958)

        elif msg.get_type() == "SYS_STATUS":
            batt_list.append(msg.voltage_battery / 1000.0)

        # trim
        for arr in [t_list, roll_list, pitch_list, yaw_list, batt_list]:
            if len(arr) > MAX_POINTS:
                arr[:] = arr[-MAX_POINTS:]

# Start thread
threading.Thread(target=mav_reader, daemon=True).start()

# ======== TIMER: UPDATE UI ========
def update_ui():
    if len(t_list) == 0:
        return

    curve_roll.setData(t_list, roll_list)
    curve_pitch.setData(t_list, pitch_list)
    curve_yaw.setData(t_list, yaw_list)

    if len(batt_list):
        curve_batt.setData(t_list[:len(batt_list)], batt_list)

timer = QtCore.QTimer()
timer.timeout.connect(update_ui)
timer.start(16)  # 60 FPS UI only

app.exec()



