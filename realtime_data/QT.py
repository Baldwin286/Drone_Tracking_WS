from pymavlink import mavutil
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import threading
import time
from collections import deque
import pyqtgraph.opengl as gl
import numpy as np

# ======== UI SETTINGS ========
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

# ======== THREAD: READ MAVLINK ========
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

threading.Thread(target=mav_reader, daemon=True).start()

# ======== 3D QUADCOPTER ========
gl_view = gl.GLViewWidget()
gl_view.setWindowTitle('3D Quadcopter')
gl_view.opts['distance'] = 6
gl_view.setBackgroundColor((50,50,50,255))
gl_view.show()

grid = gl.GLGridItem()
grid.scale(1,1,1)
gl_view.addItem(grid)

body_x, body_y, body_z = 0.3, 0.15, 0.1
body_verts = np.array([
    [-body_x/2, -body_y/2, -body_z/2],
    [ body_x/2, -body_y/2, -body_z/2],
    [ body_x/2,  body_y/2, -body_z/2],
    [-body_x/2,  body_y/2, -body_z/2],
    [-body_x/2, -body_y/2,  body_z/2],
    [ body_x/2, -body_y/2,  body_z/2],
    [ body_x/2,  body_y/2,  body_z/2],
    [-body_x/2,  body_y/2,  body_z/2],
])
faces = np.array([
    [0,1,2],[0,2,3],[4,5,6],[4,6,7],
    [0,1,5],[0,5,4],[2,3,7],[2,7,6],
    [1,2,6],[1,6,5],[0,3,7],[0,7,4]
])
colors = np.array([[1,0,0,1]]*12)
body = gl.GLMeshItem(vertexes=body_verts, faces=faces, faceColors=colors,
                     smooth=False, drawEdges=True, edgeColor=(1,1,1,1))
gl_view.addItem(body)

arms = []
arm_length = 1.0
arm_thick = 0.05
arm_colors = [(0,0,1,1),(0,1,0,1),(1,0,1,1),(1,1,0,1)]
arm_positions = [(1,0),(-1,0),(0,1),(0,-1)]
for idx, (dx, dy) in enumerate(arm_positions):
    verts = np.array([
        [0, -arm_thick/2, -arm_thick/2],
        [arm_length*dx, -arm_thick/2, -arm_thick/2],
        [arm_length*dx, arm_thick/2, -arm_thick/2],
        [0, arm_thick/2, -arm_thick/2],
        [0, -arm_thick/2, arm_thick/2],
        [arm_length*dx, -arm_thick/2, arm_thick/2],
        [arm_length*dx, arm_thick/2, arm_thick/2],
        [0, arm_thick/2, arm_thick/2],
    ])
    arm = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=np.array([arm_colors[idx]]*12),
                        smooth=False, drawEdges=True, edgeColor=(1,1,1,1))
    gl_view.addItem(arm)
    arms.append(arm)

# ======== TIMER: UPDATE UI & 3D ========
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

    if roll_list and pitch_list and yaw_list:
        roll = np.radians(roll_list[-1])
        pitch = np.radians(pitch_list[-1])  
        yaw = np.radians(yaw_list[-1])  

        cz, sz = np.cos(yaw), np.sin(yaw)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cx, sx = np.cos(roll), np.sin(roll)

        R = np.array([
            [cz*cy, cz*sy*sx + sz*cx, cz*sy*cx - sz*sx],  
            [sz*cy, sz*sy*sx - cz*cx, sz*sy*cx + cz*sx],  
            [-sy, cy*sx, cy*cx] 
        ])

        body.setMeshData(vertexes=np.dot(body_verts, R.T), faces=faces, faceColors=colors)

        for idx, (dx, dy) in enumerate(arm_positions):
            verts = np.array([
                [0, -arm_thick/2, -arm_thick/2],
                [arm_length*dx, -arm_thick/2, -arm_thick/2],
                [arm_length*dx, arm_thick/2, -arm_thick/2],
                [0, arm_thick/2, -arm_thick/2],
                [0, -arm_thick/2, arm_thick/2],
                [arm_length*dx, -arm_thick/2, arm_thick/2],
                [arm_length*dx, arm_thick/2, arm_thick/2],
                [0, arm_thick/2, arm_thick/2],
            ])
            arms[idx].setMeshData(vertexes=np.dot(verts, R.T),
                                  faces=faces,
                                  faceColors=np.array([arm_colors[idx]]*12))

# ======== START TIMER ========
timer = QtCore.QTimer()
timer.timeout.connect(update_ui)
timer.start(16)

# ======== START APP ========
app.exec()
