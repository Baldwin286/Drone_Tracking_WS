import socket
import time
import threading
from pymavlink import mavutil

mav = mavutil.mavlink_connection('/dev/ttyAMA0', baud=921600)
print("Waiting for heartbeat...")
mav.wait_heartbeat()
print("Connected!")

SERVER_IP = '0.0.0.0' 
SERVER_PORT = 5000
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_IP, SERVER_PORT))
server_socket.listen(1)
print(f"Server đang lắng nghe trên {SERVER_IP}:{SERVER_PORT}...")

def send_data(client_socket):
    while True:
        msg = mav.recv_match(type=["ATTITUDE", "SYS_STATUS"], blocking=True)
        if msg.get_type() == "ATTITUDE":
            data = {
                'type': 'ATTITUDE',
                'roll': msg.roll * 57.2958,  
                'pitch': msg.pitch * 57.2958,
                'yaw': msg.yaw * 57.2958
            }
        elif msg.get_type() == "SYS_STATUS":
            data = {
                'type': 'SYS_STATUS',
                'battery': msg.voltage_battery / 1000.0  
            }
        
        try:
            client_socket.sendall((str(data) + "\n").encode('utf-8'))
        except socket.error:
            break
        time.sleep(0.1)

def start_server():
    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Đã kết nối với {client_address}")
        threading.Thread(target=send_data, args=(client_socket,), daemon=True).start()

start_server()
