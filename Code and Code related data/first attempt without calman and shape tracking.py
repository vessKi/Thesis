#Author Kilian Vesshoff 15.02.2024

import cv2  # openCV
import numpy as np  # math
import serial  # for arduino
import time  # for waiting
import threading
import queue
from pynput import keyboard
import random # for shape selection



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
should_draw_shapes = False  # Global flag to trigger shape drawing




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
   # print("Sending: ", command.strip(), "to sender")  # For debugging
    
def pen_up():
    send_gcode("M3 S50\n")  # Pen up command
    print("Pen lifted")

def pen_down():
    send_gcode("M3 S30\n")  # Pen down command for drawing
    print("Pen lowered")

def draw_last_shape(shapes):
    if shapes:  # Check if there are any recorded shapes
        last_shape = random.choice(shapes)  # Select the last recorded shape
        print("Drawing the last shape with points:", last_shape)
        pen_up()
        start_x, start_y = camera_to_plotter(*last_shape[0])
        send_gcode(f'G0 X{start_x:.2f} Y{start_y:.2f} F10000\n')  # Move to start position without drawing
        pen_down()
        for point in last_shape:
            x_plot, y_plot = camera_to_plotter(*point)
            gcode = generate_gcode(x_plot, y_plot)
            send_gcode(gcode)
        pen_up()
        shapes.remove(last_shape)

# tryting to smooth shapes pre drawing 
def smooth_shape_with_moving_average(shape, window_size=3):
    if len(shape) < window_size or window_size < 3:
        print("Shape is too short or window size is too small.")
        return shape
    
    # Helper function to perform moving average on a list
    def moving_average(a, n=window_size):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    # Separate the shape into X and Y components
    x, y = zip(*shape)
    x = np.array(x)
    y = np.array(y)

    # Apply moving average to both X and Y components
    x_smoothed = moving_average(x, window_size)
    y_smoothed = moving_average(y, window_size)

    # Adjust the start and end points to maintain the original shape's length
    x_padded = np.pad(x_smoothed, (window_size//2, window_size//2 - 1), mode='edge')
    y_padded = np.pad(y_smoothed, (window_size//2, window_size//2 - 1), mode='edge')

    # Combine the smoothed components back into the shape format
    smoothed_shape = list(zip(x_padded, y_padded))
    return smoothed_shape
              
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

is_key_pressed = False  # Flag to track the state of the space key

def on_press(key):
    global shape_recording_mode, current_shape, is_key_pressed
    if key == keyboard.Key.space and not is_key_pressed:
        shape_recording_mode = True
        is_key_pressed = True  # Set the flag to True when recording starts
        current_shape = []  # Ensure a new shape list is ready for recording
        print("Shape recording started.")
        
def on_release(key):
    global shape_recording_mode, shapes, should_draw_shapes, current_shape, is_key_pressed
    if key == keyboard.Key.space:
        if is_key_pressed:  # Only stop recording if it was previously started
            shape_recording_mode = False
            is_key_pressed = False  # Reset the flag when the key is released
            if current_shape:  # Check if there's anything recorded
                # Apply smoothing to the current_shape
                smoothed_shape = smooth_shape_with_moving_average(current_shape, window_size=5)
                shapes.append(smoothed_shape)
                print("Shape recorded and smoothed.")
                # shapes.append(current_shape)
                # print("Shape recorded.")
            current_shape = []  # Prepare for the next shape if any subsequent recordings are made 
    elif key == keyboard.KeyCode.from_char('d'):
        if not shape_recording_mode and shapes:  # Ensure there are shapes and we're not in recording mode
            draw_last_shape(shapes)  # Call the updated function
            should_draw_shapes = False  # Reset the flag (might be redundant depending on your design)
        
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
# issue with the screen freezing cause im recording
# Initialization of the pynput listener should be done once, outside the main loop
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: frame.")
        break

    contours, mask = detect_pen(frame)

    if contours and shape_recording_mode:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                current_shape.append((cX, cY))
                x_plot, y_plot = camera_to_plotter(cX, cY)

                # Visual feedback on pen
                cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(frame, "Pen", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if should_draw_shapes and not shape_recording_mode:
        print(f"Shapes to draw: {shapes}")
        draw_last_shape(shapes)
        should_draw_shapes = False

    

                

    
    # # Detect the green ball (plotter position)
    # green_ball_contours, _ = detect_plotter(frame)
    # if green_ball_contours:
    #     # Find the largest contour by area
    #     largest_contour = max(green_ball_contours, key=cv2.contourArea)
    #     if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
    #         # Calculate the centroid of the contour
    #         M = cv2.moments(largest_contour)
    #         if M["m00"] != 0:
    #             green_cX = int(M["m10"] / M["m00"])
    #             green_cY = int(M["m01"] / M["m00"])
    #             cv2.circle(frame, (green_cX, green_cY), 7, (0, 255, 0), -1)
    #             cv2.putText(frame, "Plotter", (green_cX - 20, green_cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
    # # Draw the centre on the frame for visualization
    # if cX is not None and cY is not None:
    #     cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)

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
time.sleep(1)
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
