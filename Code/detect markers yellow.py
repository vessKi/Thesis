import cv2
import numpy as np
lower_yellow = np.array([7, 134, 116])
upper_yellow = np.array([54, 199, 255])
cap = cv2.VideoCapture(0)
    

def find_yellow_markers(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Optional: apply some morphological operations to clean up the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))
    
    return centers

def detect_markers_and_calibrate(frame):
    # Assuming find_yellow_markers() detects and returns the centers of all yellow markers in the frame
    marker_positions = find_yellow_markers(frame)

    if len(marker_positions) < 4:
        print("Error: Not all markers detected.")
        return None, None

    # Sort markers based on their y-coordinate to separate top and bottom markers
    top_markers = sorted(marker_positions, key=lambda x: x[1])[:2]
    bottom_markers = sorted(marker_positions, key=lambda x: x[1])[2:]

    # Further sort top markers to identify left and right
    top_left, top_right = sorted(top_markers, key=lambda x: x[0])
    bottom_left, bottom_right = sorted(bottom_markers, key=lambda x: x[0])

    # Define the plotter's drawing area in plotter coordinates
    # Here we assume the bottom_left marker is the origin (0,0) of the plotter's coordinate system
    # Adjust according to your setup, especially if the origin is offset from the bottom_left marker
    plotter_points = np.array([
        [0, 0],  # Bottom-left as origin
        [PLOTTER_WIDTH, 0],  # Top-right
        [0, PLOTTER_HEIGHT]  # Bottom-left to top-left vertical displacement for height
    ], dtype="float32")

    # Camera points collected from the detected markers
    camera_points = np.array([bottom_left, top_right, top_left], dtype="float32")

    # Calculate the transformation matrix from camera to plotter coordinates
    transformation_matrix = cv2.getAffineTransform(camera_points, plotter_points)

    return transformation_matrix, bottom_left

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for yellow color
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Optional: apply some morphological operations to clean up the mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the original frame
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        # Display the original frame and the mask
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
def main():
    
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        cap.release()
    else:
        transformation_matrix, plotter_origin = detect_markers_and_calibrate(frame)
        if transformation_matrix is not None and not np.all((transformation_matrix == 0)):
            print("Calibration Successful. Transformation Matrix:", transformation_matrix)
            print("Plotter Origin in Camera Frame:", plotter_origin)
        else:
            print("Calibration Failed.")
            cap.release()
            cv2.destroyAllWindows()
            
        cap.release()
if __name__ == "__main__":
    main()
