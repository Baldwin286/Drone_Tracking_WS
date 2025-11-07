from picamera2 import Picamera2
import cv2
import time

def main():
    print("Initializing Pi Camera V3...")
    picam2 = Picamera2()

    # Create preview configuration (adjust size if needed)
    config = picam2.create_preview_configuration(
        main={"size": (640, 360), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    print("Pi Camera started successfully!")
    print("Press 'q' to quit.")

    prev_time = time.time()
    fps = 0.0

    while True:
        frame = picam2.capture_array()
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # Show FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Pi Camera V3 Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Stopping camera...")
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera test ended.")

if __name__ == "__main__":
    main()