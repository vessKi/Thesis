import cv2
import numpy as np
import json
import os

# Define constants for A4 paper size in mm (adjust based on your setup)
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
A4_ASPECT_RATIO = A4_WIDTH_MM / A4_HEIGHT_MM

# Load calibration data
with np.load('c:/Users/kilia/Desktop/COOP/Thesis/calibration_data.npz') as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

# File to save/load corner coordinates
corners_file = 'corners.json'

# List to store the corner points
corner_points = []

def save_corners(corners):
    with open(corners_file, 'w') as f:
        json.dump(corners, f)

def click_event(event, x, y, flags, param):
    global corner_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corner_points) < 4:
            corner_points.append((x, y))
            print(f"Corner {len(corner_points)}: ({x}, {y})")
        if len(corner_points) == 4:
            save_corners(corner_points)

def detect_paper(frame, corners):
    if corners:
        for point in corners:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
        return np.array(corners, dtype="float32")
    return None

def main():
    global corner_points
    corner_points = []  # Clear the corner points each time the program starts

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use the appropriate camera index (0 for default)
    desired_width = 1280
    desired_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort the frame
        frame_undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
        
        # Detect paper using the saved or selected corners
        paper_corners = detect_paper(frame_undistorted, corner_points)
        
        if paper_corners is not None:
            cv2.polylines(frame_undistorted, [np.int32(paper_corners)], isClosed=True, color=(0, 255, 0), thickness=2)
        else:
            cv2.putText(frame_undistorted, "Click to select corners", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Paper Detection', frame_undistorted)
        cv2.setMouseCallback('Paper Detection', click_event)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
