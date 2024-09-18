import cv2
import time

def find_available_cameras(limit=10):
    available_cameras = []
    for i in range(limit):
        print(f"Checking camera at index {i}...")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Adding cv2.CAP_DSHOW for Windows
        time.sleep(0.1)  # Short wait time to ensure isOpened() has time to update its status
        if cap.isOpened():
            print(f"Camera found at index {i}!")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"No camera found at index {i}.")
    return available_cameras

available_cameras = find_available_cameras()
print("Available cameras are at indices:", available_cameras)
