#Author Kilian Vesshoff 15.02.2024

import cv2  # openCV
import numpy as np  # math
import serial  # for arduino
import time  # for waiting
import threading
import queue

# Open serial connection to Plotter
ser = serial.Serial('COM4', 115200) 
if not ser.isOpen():
    ser.open()
print("Plotter Online")
soft_reset_command = b'\x18'  # GRBL soft-reset command
ser.write(soft_reset_command)
time.sleep(0.2)
ser.flushInput()  # Clear any data from the input buffer
ser.flushOutput()  # Clear any data from the output buffer
print("Plotter Awake and Listening")
time.sleep(0.2)
# Create a queue for GCode commands
gcode_queue = queue.Queue()

# Initialize stuff for later
last_x_plot, last_y_plot = None, None
last_send_time = time.time()
distance_threshold = 10
send_interval = 0.2
commands_waiting = 0
#shape detection
shape_recording_mode = False
shapes = []  # List to hold lists of points for each shape
current_shape = []  # List to hold points of the currently drawn shape




# Plotter surface is 445x412 new webcam 720p
def camera_to_plotter(x_cam, y_cam, cam_resolution=(1280, 720), plotter_size=(840, 550)):
    x_scale = plotter_size[0] / cam_resolution[0]
    y_scale = plotter_size[1] / cam_resolution[1]
    x_plot = x_cam * x_scale #flipped
    y_plot = y_cam * y_scale
    return x_plot, y_plot


# Function to generate G-code for the plotter
def generate_gcode(x_plot, y_plot, feed_rate=10000):
    # Include the feed rate in each move command
    move_command = f"G01 X{x_plot:.2f} Y{y_plot:.2f} F{feed_rate}"  # Now setting the feed rate here
    return f"{move_command}\n"


def gcode_sender():
    global commands_waiting  # Declare commands_waiting as global
    while True:
        command = gcode_queue.get()  # Wait until a command is available
        if command == "QUIT":
            break  # Exit loop if "QUIT" command is received
        print("Sending: ", command.strip())
        ser.write(command.encode())
        commands_waiting += 1  # Increment when a command is sent
        # Wait for a response from the plotter
        response = ser.readline().decode().strip()
        print("Plotter response: ", response)
        # Check for acknowledgments to decrement the counter
        if response == 'ok' or response.startswith('error'):
            commands_waiting -= 1
        # Additionally, check if the response contains buffer information
        if response.startswith('<') and response.endswith('>'):
            parts = response.split('|')
            for part in parts:
                if part.startswith('Bf:'):
                    buffer_info = part[3:].split(',')
                    print(f"Buffer space: {buffer_info[0]}, RX buffer: {buffer_info[1]}")
        
        gcode_queue.task_done()  # Mark the task as done

# Start the GCode sender thread
sender_thread = threading.Thread(target=gcode_sender, daemon=True)
sender_thread.start()
            
#sending codes
def send_gcode(command):
    gcode_queue.put(command)
    print("Sending: ", command.strip(), "to sender")  # For debugging
    
def pen_up():
    send_gcode("M3 S50\n")  # Pen up command
    print("Pen lifted")

def pen_down():
    send_gcode("M3 S30\n")  # Pen down command for drawing
    print("Pen lowered")
            
# Function to detect the pen using color segmentation
def detect_pen(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #detect white 
    #lower_color = np.array([0, 0, 220])  # Low saturation, high value
    #upper_color = np.array([180, 255, 255])  # Full range of hue, higher range of saturation, maximum value
    #detect red
    lower_color = np.array([0, 120, 70])
    upper_color = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Noise reduction
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

# Function to detect the green ball using color segmentation
def detect_plotter(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define the color range for detecting green. Adjust these values.
    lower_green = np.array([40, 40, 40])  # Example range, adjust based on your green ball
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Noise reduction
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

# Initialization camera
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
print("CamOnline and running")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
pen_up()
if not cap.isOpened():
    print("Error: cam.")
    ser.close()
    exit()

# Main processing loop while camera is online
frame_counter = 0  # Initialize frame counter
cX = None
cY = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: frame.")
        break

    contours, mask = detect_pen(frame)
    #shape detection
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # If the space bar is pressed
        shape_recording_mode = not shape_recording_mode  # Toggle shape recording mode
        if not shape_recording_mode:
            # Exiting shape recording mode, save the current shape
            if current_shape:
                shapes.append(current_shape)
                current_shape = []  # Reset for the next shape
    #contours
    if contours and shape_recording_mode:
        # Find the largest contour by area and detect shape
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
            # Calculate the centroid of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                current_shape.append((cX, cY))  # Append the current point to the shape
                 # Convert camera coordinates to plotter coordinates
                x_plot, y_plot = camera_to_plotter(cX, cY)
            
            # Check if there's a significant movement
                if last_x_plot is None or last_y_plot is None or abs(x_plot - last_x_plot) > 10 or abs(y_plot - last_y_plot) > 10:
                    gcode = generate_gcode(x_plot, y_plot)
                    send_gcode(gcode)
                    last_x_plot, last_y_plot = x_plot, y_plot

                # Here you can integrate the plotting logic as before
                # For example, converting (cX, cY) to plotter coordinates and sending G-code

                # Visual feedback
                
                cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(frame, "Pen", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                

    
    # Detect the green ball (plotter position)
    green_ball_contours, _ = detect_plotter(frame)
    if green_ball_contours:
        # Find the largest contour by area
        largest_contour = max(green_ball_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
            # Calculate the centroid of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                green_cX = int(M["m10"] / M["m00"])
                green_cY = int(M["m01"] / M["m00"])
                cv2.circle(frame, (green_cX, green_cY), 7, (0, 255, 0), -1)
                cv2.putText(frame, "Plotter", (green_cX - 20, green_cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
    # Draw the centre on the frame for visualization
    if cX is not None and cY is not None:
        cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)

    # Display the updated frame
    cv2.imshow('Master Thesis Vessi', frame)

    # Break the loop if 'q' is pressed or x is clicked
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Master Thesis Vessi', cv2.WND_PROP_VISIBLE) < 1:
        break

    # Increment frame counter
    frame_counter += 1

    
# Simplified cleanup procedure
print("Initiating cleanup...")


# Step 1: Soft-reset GRBL to ensure it aborts any ongoing operations and clears its buffer
soft_reset_command = b'\x18'  # GRBL soft-reset command
ser.write(soft_reset_command)
time.sleep(0.5)  # Give GRBL a moment to process the reset; adjust time as needed


# Step 2: Clear the serial buffers
ser.flushInput()  # Clear any data in the input buffer
ser.flushOutput()  # Clear any data in the output buffer
print("GRBL buffers cleared.")
time.sleep(1)

 # Step 2.5: Move plotter back to origin after homing after pen up
print("Moving to origin (0,0)...")
pen_up()
ser.write(b'G90\n')  # Ensure absolute positioning mode
ser.write(b'G0 X0 Y0\n')  # Move to origin
time.sleep(5)  # Adjust this delay based on your machine's movement speed and size
pen_down()
time.sleep(1)


# Step 3: Close the serial connection
ser.close()
print("Serial connection closed.")

# Step 4: Release camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
gcode_queue.put("QUIT")
sender_thread.join() 
print("Camera released and windows closed. Cleanup complete.")
