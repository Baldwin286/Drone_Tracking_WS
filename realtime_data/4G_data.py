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

try:
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(1)
    print(f"Server hear on {SERVER_IP}:{SERVER_PORT}...")
except socket.error as e:
    print(f"Error when bind server: {e}")
    exit(1) 

def send_data(client_socket):
    print("Thread started for sending data.")
    while True:
        try:
            msg = mav.recv_match(type=["ATTITUDE", "SYS_STATUS"], blocking=True)
            print(f"Received message: {msg}")  

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

            print("SEND:", data)
            client_socket.sendall((str(data) + "\n").encode('utf-8'))
        except socket.error as e:
            print(f"Error when send data through socket: {e}")
            break
        except Exception as e:
            print(f"Error...: {e}")
        time.sleep(0.1)

def start_server():
    while True:
        try:
            client_socket, client_address = server_socket.accept()
            print(f"Connected to {client_address}")
            threading.Thread(target=send_data, args=(client_socket,), daemon=True).start()
        except Exception as e:
            print(f"Error when connect to client: {e}")
            continue 

try:
    start_server()
except KeyboardInterrupt:
    print("Server stopped by User")
except Exception as e:
    print(f"Error when run server: {e}")
