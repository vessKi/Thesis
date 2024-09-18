#Author Kilian Vesshoff 15.02.2024

import cv2 #openCV
import numpy as np #math
import serial # for arduino
import time #for waiting

# Open serial connection to Plotter
# Lower USB is com4 near "TAB"
ser = serial.Serial('COM4', 115200)  # Adjust COM_PORT to your setup
print("Plotter Online")
time.sleep(0.1)  #Wait a bit
print("Plotter Awake and Listening")
ser.write(('F9000' + '\n').encode())
time.sleep(0.1)  

#inizalize stuff for later
last_x_plot, last_y_plot = None, None
last_send_time = time.time()
distance_threshold = 20
send_interval = 0.5

# Initialize the Kalman Filter (hope this works)
kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # Measurement matrix
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)  # Transition matrix
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Process noise
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10  # Measurement noise

#Kalman Filter function
def update_kalman_filter(kalman, measured_x, measured_y):
    # Prediction
    predicted = kalman.predict()
    
    # Measurement Update
    measurement = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])
    estimated = kalman.correct(measurement)
    
    return estimated[0], estimated[1]  # Extract the estimated x, y position

#plotter surface is 445x412 webcam is needs to have a higher res as seen below
def camera_to_plotter(x_cam, y_cam, cam_resolution=(1920, 1080), plotter_size=(150, 100)): # set up plotter size in whatever - 5cm 
    #include a "rahmen" for the plotter to reduce the chance of it hitting the edges 
    x_scale = plotter_size[0] / cam_resolution[0]
    y_scale = plotter_size[1] / cam_resolution[1]
    x_plot = x_cam * x_scale
    y_plot = y_cam * y_scale
    return x_plot, y_plot

# Function to generate G-code for the plotter
def generate_gcode(x_plot, y_plot, pen_down=True):
    pen_command = "M3 S15\n" if pen_down else "M3 S50\n"  # Adjust pen up/down commands as per your setup
    move_command = "G1 X{} Y{}".format(x_plot, y_plot)
    full_command = "{}\n{}".format(pen_command, move_command)
    return full_command


# Function to send G-code to the plotter
def send_gcode(command):
    ser.write(('F5000' + '\n').encode())
    print("Sending: " + command)  # For debugging
    ser.write((command + '\n').encode())
    time.sleep(0.1)  # Adjust based on trial and error for your specific setup
    ser.write((command + '\n').encode())
        
        
def detect_red_ball(frame):
    # Reduce noise, track for 4 reds 
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    # First range for red (covering the lower end of the hue spectrum)
    lower_red = np.array([35, 50, 40])
    upper_red = np.array([75, 255, 255])
    # Second range for red (covering the higher end of the hue spectrum)
    lower_red2 = np.array([40, 100, 100])
    upper_red2 = np.array([90, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

def get_initial_tracking_point(contours):
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        return np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)  # Format required by calcOpticalFlowPyrLK
    return None

 # Initialization camera
#stream_url = 'http://192.168.0.26:8080/video'
# Create a VideoCapture object
cap = cv2.VideoCapture(0)
print("CamOnline and running")
if not cap.isOpened():
    print("Error: cam.")
    ser.close()
    exit()

# need these for sending gcode
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
last_x_plot, last_y_plot = None, None
last_send_time = time.time()
pen_down = False

# Main processing loop while camera is online
frame_counter = 0  # Initialize frame counter
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: frame.")
        break

    # Process the frame for red ball detection
    cX, cY = None, None
    contours, mask = detect_red_ball(frame)
    if contours and frame_counter % 5 == 0:  # Process for G-code sending only every 5 frames
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        # Calculate the centroid of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Update Kalman filter with the detected position
            kalman.correct(np.array([[np.float32(cX)], [np.float32(cY)]]))
            # Get the Kalman prediction
            prediction = kalman.predict()
            x_pred, y_pred = int(prediction[0][0]), int(prediction[1][0])

            # Convert camera coordinates to plotter coordinates
            x_plot, y_plot = camera_to_plotter(x_pred, y_pred)

            # Check if significant movement detected or time interval passed
            current_time = time.time()
            if last_x_plot is None or np.sqrt((x_plot - last_x_plot) ** 2 + (y_plot - last_y_plot) ** 2) > distance_threshold or (current_time - last_send_time > send_interval):
                gcode = generate_gcode(x_plot, y_plot, pen_down=True)
                send_gcode(gcode)
                last_x_plot, last_y_plot = x_plot, y_plot
                last_send_time = current_time

    # Draw the centroid on the frame for visualization
    if cX is not None and cY is not None:
        cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)

    # Display the updated frame
    cv2.imshow('Master Thesis Vessi', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Master Thesis Vessi', cv2.WND_PROP_VISIBLE) < 1:
        break

    # Increment frame counter
    frame_counter += 1


    
# Simplified cleanup procedure
print("Initiating cleanup...")

# Step 1: Soft-reset GRBL to ensure it aborts any ongoing operations and clears its buffer
soft_reset_command = b'\x18'  # GRBL soft-reset command
ser.write(soft_reset_command)
time.sleep(1)  # Give GRBL a moment to process the reset; adjust time as needed

# Step 2: Clear the serial buffers
ser.flushInput()  # Clear any data in the input buffer
ser.flushOutput()  # Clear any data in the output buffer
print("GRBL buffers cleared.")

# Step 3: Close the serial connection
ser.close()
print("Serial connection closed.")

# Step 4: Release camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed. Cleanup complete.")
