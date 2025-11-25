import socket
import time
import threading
from pymavlink import mavutil

# Kết nối MAVLink với flight controller
mav = mavutil.mavlink_connection('/dev/ttyAMA0', baud=921600)
print("Waiting for heartbeat...")
mav.wait_heartbeat()
print("Connected!")

# Thiết lập server TCP
SERVER_IP = '100.97.30.93'  # Lắng nghe trên tất cả các giao diện
SERVER_PORT = 5000
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(1)
    print(f"Server đang lắng nghe trên {SERVER_IP}:{SERVER_PORT}...")
except socket.error as e:
    print(f"Lỗi khi bind server: {e}")
    exit(1)  # Dừng chương trình nếu không thể bind

# Hàm gửi dữ liệu tới client
def send_data(client_socket):
    print("Thread started for sending data.")
    while True:
        try:
            msg = mav.recv_match(type=["ATTITUDE", "SYS_STATUS"], blocking=True)
            print(f"Received message: {msg}")  # In ra thông báo để kiểm tra dữ liệu nhận được

            if msg.get_type() == "ATTITUDE":
                data = {
                    'type': 'ATTITUDE',
                    'roll': msg.roll * 57.2958,  # Chuyển từ radian sang độ
                    'pitch': msg.pitch * 57.2958,
                    'yaw': msg.yaw * 57.2958
                }
            elif msg.get_type() == "SYS_STATUS":
                data = {
                    'type': 'SYS_STATUS',
                    'battery': msg.voltage_battery / 1000.0  # Chuyển từ mV sang V
                }

            # In dữ liệu trước khi gửi đi
            print("SEND:", data)
            client_socket.sendall((str(data) + "\n").encode('utf-8'))
        except socket.error as e:
            print(f"Lỗi khi gửi dữ liệu qua socket: {e}")
            break
        except Exception as e:
            print(f"Lỗi khác: {e}")
        time.sleep(0.1)

# Hàm bắt đầu server và chấp nhận kết nối từ client
def start_server():
    while True:
        try:
            client_socket, client_address = server_socket.accept()
            print(f"Đã kết nối với {client_address}")
            threading.Thread(target=send_data, args=(client_socket,), daemon=True).start()
        except Exception as e:
            print(f"Lỗi khi kết nối client: {e}")
            continue  # Tiếp tục chạy server ngay cả khi có lỗi kết nối

# Chạy server
try:
    start_server()
except KeyboardInterrupt:
    print("Server bị dừng bởi người dùng.")
except Exception as e:
    print(f"Lỗi bất ngờ khi chạy server: {e}")
