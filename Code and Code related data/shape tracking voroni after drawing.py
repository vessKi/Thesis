#Author Kilian Vesshoff 15.02.2024

import cv2  # openCV
import numpy as np  # math
import serial  # for arduino
import time  # for waiting
import threading 
import queue
from pynput import keyboard
import random # for shape selection
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


# Open serial connection to Plotter
ser = serial.Serial('COM4', 115200) 
PLOTTER_WIDTH = 300
PLOTTER_HEIGHT = 300
gcode_queue = queue.Queue() # for gcode send

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
def camera_to_plotter(x_cam, y_cam, cam_resolution=(1280, 720)):
    plotter_size = (PLOTTER_WIDTH, PLOTTER_HEIGHT)
    x_scale = plotter_size[0] / cam_resolution[0]
    y_scale = plotter_size[1] / cam_resolution[1]
    x_plot = max(min(x_cam * x_scale, PLOTTER_WIDTH - 1), 0)  # Ensure within bounds
    y_plot = max(min(y_cam * y_scale, PLOTTER_HEIGHT - 1), 0)
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

#sending codes
def send_gcode(command):
    gcode_queue.put(command)
   # print("Sending: ", command.strip(), "to sender")  # For debugging
    
def pen_up():
    send_gcode("M3 S65\n")  # Pen up command
    print("Pen lifted")

def pen_down():
    send_gcode("M3 S25\n")  # Pen down command for drawing
    print("Pen lowered")

def detect_paper_edges_and_set_origin(frame):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edged = cv2.Canny(blurred, 50, 150)
    
    # Find contours and sort by size
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Assume the largest rectangular contour is the paper
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            # Transform coordinates such that the top-left corner of the detected paper is (0,0)
            transformed = transform_coordinates_to_origin(approx)
            return transformed
    return None


def transform_coordinates_to_origin(shapes, top_left):
    # 'shapes' is a list of shapes with their coordinates
    # 'top_left' is the top-left corner of the paper detected earlier
    transformed_shapes = []
    for shape in shapes:
        transformed_shape = [(point[0] - top_left[0], point[1] - top_left[1]) for point in shape]
        transformed_shapes.append(transformed_shape)
    return transformed_shapes

def plot_voronoi_diagram(points):
    # Generate the Voronoi diagram
    vor = Voronoi(points)
    # Plot using Matplotlib
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
    # Customizations
    ax.set_xlim([0, PLOTTER_WIDTH])  # Adjust the x-axis limits
    ax.set_ylim([0, PLOTTER_HEIGHT])  # Adjust the y-axis limits
    ax.set_title("Voronoi Diagram")

    plt.show()


    
def draw_voronoi(shapes):
    # Ensure points are within plotter bounds before generating Voronoi diagram
    bounded_points = [camera_to_plotter(pt[0], pt[1], cam_resolution=(1280, 720)) for shape in shapes for pt in shape]
    all_points = np.array(bounded_points)
    
    if len(all_points) < 3:
        print("Not enough points for Voronoi.")
        return
    plot_voronoi_diagram(all_points)

    vor = Voronoi(all_points)
    
    # Draw edges ensuring they are within bounds
    for edge in vor.ridge_vertices:
        start, end = edge
        if start >= 0 and end >= 0:
            start_pt = vor.vertices[start]
            end_pt = vor.vertices[end]
            if is_point_within_bounds(start_pt) and is_point_within_bounds(end_pt):
                draw_line(start_pt, end_pt)

def is_point_within_bounds(point):
    x, y = point
    return 0 <= x <= PLOTTER_WIDTH and 0 <= y <= PLOTTER_HEIGHT

def draw_line(start_pt, end_pt):
    # Convert Voronoi vertices directly
    x1, y1 = start_pt
    x2, y2 = end_pt
    # Clip to ensure within plotter bounds
    x1, y1 = np.clip(x1, 0, PLOTTER_WIDTH), np.clip(y1, 0, PLOTTER_HEIGHT)
    x2, y2 = np.clip(x2, 0, PLOTTER_WIDTH), np.clip(y2, 0, PLOTTER_HEIGHT)
    send_gcode(generate_gcode(x1, y1))  # Move to start
    pen_down()
    send_gcode(generate_gcode(x2, y2))  # Draw to end
    pen_up()

def draw_shape(shape):
    pen_up()
    if not shape:  # Check if the shape is empty
        return
    start_x, start_y = camera_to_plotter(*shape[0])
    send_gcode(f'G0 X{start_x:.2f} Y{start_y:.2f} F10000\n')
    pen_down()
    for point in shape[1:]:
        x_plot, y_plot = camera_to_plotter(*point)
        send_gcode(generate_gcode(x_plot, y_plot))
    pen_up()

def find_paper_edges(frame):
    print("Finding paper edges...")
    return np.array([[[0, 0]], [[0, 100]], [[100, 100]], [[100, 0]]])

def find_top_left_corner(corners):
    print("Finding top left corner...")
    return corners[0][0]

def detect_shapes(frame):
    print("Detecting shapes...")
    return np.array([np.array([[10, 10], [10, 20], [20, 20], [20, 10]])])

def plot_shapes(shapes):
    print("Plotting shapes...")
    # Simplified plotting for demonstration
    plt.plot(shapes[:, :, 0], shapes[:, :, 1], 'r-')
    plt.show()
    
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
frame_queue = queue.Queue(maxsize=5)  # Adjust size as needed

def detect_pen(frame):
    while True:
        frame = frame_queue.get()
        if frame is None:  # Use None as a signal to stop the thread
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Adjusted HSV range for a typical green pen
        lower_color = np.array([36, 25, 25])  # Lower bound of green hue range
        upper_color = np.array([86, 255,255])  # Upper bound of green hue range
        mask = cv2.inRange(hsv, lower_color, upper_color)
        # Noise reduction
        mask = cv2.erode(mask, None, iterations=5)
        mask = cv2.dilate(mask, None, iterations=5)
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, mask

is_key_pressed = False  # Flag to track the state of the space key

def on_press(key):
    if key == keyboard.Key.space:
        print("Space pressed, executing functions...")
        # Assuming the frame is captured globally or from your camera capture logic
        frame = np.zeros((100, 100, 3), dtype=np.uint8)  # Dummy frame for demonstration
        
        # Replace the following function calls with your actual logic
        paper_edges = find_paper_edges(frame)
        if paper_edges is not None:
            top_left_corner = find_top_left_corner(paper_edges)
            shapes = detect_shapes(frame)  # This needs to be your actual shape detection logic
            plot_shapes(shapes)
        else:
            print("No paper detected.")
    else:
        print(f"Key {key} pressed.")

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
         if shapes:
            draw_voronoi(shapes)
            shapes = []
# Queue to hold model predictions
predictions_queue = queue.Queue()

frame_queue = queue.Queue(maxsize=10)
detected_shapes_queue = queue.Queue()

# detect all drawn shapes on the paper
def detect_shapes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) > 3:  # Filter out non-polygon shapes if necessary
            shapes.append(approx)
    return shapes


def pen_detection_process():
    while True:
        frame = frame_queue.get()
        if frame is None:  # Stop signal
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([36, 25, 25])  # Lower bound of green hue range
        upper_color = np.array([86, 255,255])  # Upper bound of green hue range
        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours and shape_recording_mode:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    detected_shapes_queue.put((cX, cY))  # Send detected point to main thread
 
# Start the GCode sender thread
sender_thread = threading.Thread(target=gcode_sender, daemon=True)
sender_thread.start()        

# Start the pen detection thread
threading.Thread(target=pen_detection_process, daemon=True).start()
        
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
cX = None
cY = None
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Main loop adjustments
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: frame.")
        break

    if not frame_queue.full():
        frame_queue.put(frame.copy())  # Copy the frame to the queue
        
    while not detected_shapes_queue.empty():
        cX, cY = detected_shapes_queue.get()  # Retrieve detected point from the queue
        current_shape.append((cX, cY))  # Process point as before
        x_plot, y_plot = camera_to_plotter(cX, cY)
        # Draw on the frame pen
        cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Pen", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the updated frame
    cv2.imshow('Master Thesis Vessi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Master Thesis Vessi', cv2.WND_PROP_VISIBLE) < 1:
        break


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
