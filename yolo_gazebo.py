import torch
import cv2
import numpy as np
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time, sys
from ultralytics import YOLO

CONN = 'udp:127.0.0.1:14550'
print("Connecting to vehicle on:", CONN)
vehicle = connect(CONN, wait_ready=True, heartbeat_timeout=30)

model = YOLO('/home/phamthanhbien/yolo/my_model.pt')

def detect_person(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    results = model(img)  

    boxes = results[0].boxes.xyxy.cpu().numpy()  
    scores = results[0].boxes.conf.cpu().numpy()  
    classes = results[0].boxes.cls.cpu().numpy()  

    
    person_preds = [box for box, score, class_id in zip(boxes, scores, classes) if int(class_id) == 0 and score > 0.5]

    if len(person_preds) > 0:
        x1, y1, x2, y2 = person_preds[0][:4]
        cx = int((x1 + x2) / 2)  
        cy = int((y1 + y2) / 2)
    
        area = (x2 - x1) * (y2 - y1)
        
        return (cx, cy), person_preds, area
    return None, None, None

def draw_bounding_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def arm_and_takeoff(aTargetAltitude):
    print("Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    t0 = time.time()
    while not vehicle.armed:
        if time.time() - t0 > 10:
            print("Timeout arming")
            return False
        print(" Waiting for arm...")
        time.sleep(0.5)

    print("Taking off to", aTargetAltitude, "m")
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(" Altitude:", round(alt,2))
        if alt >= aTargetAltitude * 0.95:
            print("Reached target")
            break
        time.sleep(0.5)
    return True

def send_ned_velocity(vx, vy, vz, duration=1.0):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,  
        0, 0, 0,              
        vx, vy, vz,          
        0, 0, 0,               
        0, 0)
    end = time.time() + duration
    while time.time() < end:
        vehicle.send_mavlink(msg)
        vehicle.flush()
        time.sleep(0.1)

def move_towards_person(cx, cy, frame_width, frame_height, area):
    safe_distance_area = 10000  
    
    if area < safe_distance_area:
        vx = (cx - frame_width // 2) / (frame_width // 2)
        vy = (cy - frame_height // 2) / (frame_height // 2)
        vz = 0
    else:
        vx = -(cx - frame_width // 2) / (frame_width // 2)
        vy = -(cy - frame_height // 2) / (frame_height // 2)
        vz = 0
    
    send_ned_velocity(vx * 2, vy * 2, vz, duration=1.0)

if __name__ == "__main__":
    try:
        if not arm_and_takeoff(5):
            print("Arming/takeoff failed")
            vehicle.close()
            sys.exit(1)

        cap = cv2.VideoCapture(0) 

        while True:
            _, frame = cap.read()
            person_pos, person_preds, area = detect_person(frame)

            if person_pos:
                cx, cy = person_pos
                print(f"Detected person at ({cx}, {cy}), Area: {area}")
                move_towards_person(cx, cy, frame.shape[1], frame.shape[0], area)  

                frame = draw_bounding_boxes(frame, person_preds)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Returning / Land")
        vehicle.mode = VehicleMode("LAND")
        time.sleep(5)

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        vehicle.close()
        print("Closed")
