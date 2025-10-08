"""Author: Pham Thanh Bien
   Date: 2025-09-20
   Description: A simple tracking drone using Python
   Usage: python tracking_drone.py --connect dev/ttyAMA0"""
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil 
import time
import argparse
import geopy.distance
import numpy as np
import cv2
import threading

# ====== STREAM + FACE DETECTION ======

def stream_and_detect():
    width = 320
    height = 240

    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    if face_detect.empty():
        raise IOError("Unable to load the face cascade classifier xml file")

    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Camera Error!")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    print(f"Camera: {width}x{height} @ {fps}fps")

    gst_str = (
        f"appsrc ! video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
        f"videoconvert ! "
        f"x264enc tune=zerolatency bitrate=1000 speed-preset=superfast ! "
        f"rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host=192.168.1.100 port=5000"
    )

    out = cv2.VideoWriter(
        gst_str,
        cv2.CAP_GSTREAMER,
        0,
        fps,
        (width, height),
        True
    )

    if not out.isOpened():
        print("Pipeline Error!")
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

        out.write(frame)

    cap.release()
    out.release()

# ====== DRONEKIT CONNECTION ======

def connectMyCopter():
  parser =  argparse.ArgumentParser(description='commands')
  parser.add_argument('--connect')
  args = parser.parse_args()

  connection_string = args.connect
  baud_rate = 921600
  print("\nConnecting to vehicle on: %s" % connection_string)
  vehicle = connect(connection_string,baud=baud_rate,wait_ready=True)
  print("Autopilot Firmware version: %s" % vehicle.version)
  print("Autopilot capabilities (supports ftp): %s" % vehicle.capabilities.ftp)
  print("Global Location: %s" % vehicle.location.global_frame)
  print("Global Location (relative altitude): %s" % vehicle.location.global_relative_frame)
  print("Local Location: %s" % vehicle.location.local_frame)                                
  print("Attitude: %s" % vehicle.attitude)
  print("Velocity: %s" % vehicle.velocity)
  print("GPS: %s" % vehicle.gps_0) 
  print("Groundspeed: %s" % vehicle.groundspeed)
  print("Airspeed: %s" % vehicle.airspeed)
  print("Battery: %s" % vehicle.battery)
  print("EKF OK?: %s" % vehicle.ekf_ok)
  print("Last Heartbeat: %s" % vehicle.last_heartbeat)
  print("Rangefinder: %s" % vehicle.rangefinder)
  print("Rangefinder distance: %s" % vehicle.rangefinder.distance)
  print("Rangefinder voltage: %s" % vehicle.rangefinder.voltage)
  print("Heading: %s" % vehicle.heading)
  print("Is Armable?: %s" % vehicle.is_armable)
  print( "System status: %s" % vehicle.system_status.state)
  print("Mode: %s" % vehicle.mode.name)                         # settable
  print("Armed: %s" % vehicle.armed)                            # settable
  return  vehicle
  
# ====== ARM AND TAKEOFF ======

def arm_and_takeoff(vehicle, aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

        
    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:      
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)      
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95: #Trigger just below target alt
            print("Reached target altitude")
            break
        time.sleep(1)


# def get_dstance(cord1, cord2):
#     #return distance n meter
#     return (geopy.distance.geodesic(cord1, cord2).km)*1000


# #connect to drone
# vehicle = connectMyCopter()


# def goto_location(to_lat, to_long):    
        
#     print(" Global Location (relative altitude): %s" % vehicle.location.global_relative_frame)
#     curr_lat = vehicle.location.global_relative_frame.lat
#     curr_lon = vehicle.location.global_relative_frame.lon
#     curr_alt = vehicle.location.global_relative_frame.alt

#     # set to locaton (lat, lon, alt)
#     to_lat = to_lat
#     to_lon = to_long
#     to_alt = curr_alt

#     to_pont = LocationGlobalRelative(to_lat,to_lon,to_alt)
#     vehicle.simple_goto(to_pont, groundspeed=1)
    
#     to_cord = (to_lat, to_lon)
#     while True:
#         curr_lat = vehicle.location.global_relative_frame.lat
#         curr_lon = vehicle.location.global_relative_frame.lon
#         curr_cord = (curr_lat, curr_lon)
#         print("curr location: {}".format(curr_cord))
#         distance = get_dstance(curr_cord, to_cord)
#         print("distance ramaining {}".format(distance))
#         if distance <= 2:
#             print("Reached within 2 meters of target location...")
#             break
#         time.sleep(1)

# def my_mission():
#     arm_and_takeoff(5)
#     goto_location(25.806476,86.778428)
#     time.sleep(2)
#     print("Returning to Launch")
#     vehicle.mode = VehicleMode("RTL")
    
    
    
# # invoke the mission    
# my_mission()

# ====== CONNECT MY COPTER ======
vehicle = connectMyCopter()

# ====== MAIN MISSION ======
def my_mission():

    #start video stream + face detection in a separate thread
    stream_thread = threading.Thread(target=stream_and_detect, daemon=True)
    stream_thread.start()

    #takeoff to 5 meters
    arm_and_takeoff(vehicle, 5)

    print("[INFO] Hovering in place. Video streaming ongoing...")

    try:
        # Drone hover
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Mission interrupted by user.")
        vehicle.mode = VehicleMode("LAND")
        print("[INFO] Landing...")
        time.sleep(5)

if __name__ == "__main__":
    my_mission()