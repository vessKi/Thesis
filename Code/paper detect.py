import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard
import serial  # for arduino
from serial import Serial
import time  # for waiting
import threading 
import queue
from queue import Queue
from scipy.spatial import Voronoi, voronoi_plot_2d

#BIG UPDATE
# no longer translating the coordinates within the function just in the gcode generation
# finally have a somewhat accurate translation matrix with a 0,0 and a offset that is not quiete right yet.
# need to adjust every other function to use the new matrix and simplyfy the entire thing by alot
#get voroni to work and the paper detection


with np.load('calibration_data.npz') as X: # generated with the chess calibration.py and the images in the folder chess
    mtx, dist = [X[i] for i in ('mtx','dist')]

ser = serial.Serial('COM3', 115200, timeout=1)
transformation_matrix = None
gcode_queue = queue.Queue() # for gcode send

global marker_positions, scaling_factor
marker_positions = []
scaling_factor = 1.0

if not ser.isOpen():
    ser.open()
soft_reset_command = b'\x18'  # GRBL soft-reset command
ser.write(soft_reset_command)
time.sleep(0.1)
ser.flushInput()  # Clear any data from the input buffer
ser.flushOutput()  # Clear any data from the output buffer
print("Plotter Awake and Listening")
time.sleep(0.1)

# Global variable to store detected shapes
detected_shapes = []
commands_waiting = 0 
max_buffer_size = 128 
#find yellow markers on plotter
lower_bound_yellow = np.array([12, 76, 129])
upper_bound_yellow = np.array([67, 187, 255])
# in mm
REAL_DISTANCE_TOP = 394
REAL_DISTANCE_LEFT = 268
REAL_DISTANCE_DIAGONAL = 485
#in mm
PLOTTER_WIDTH = 394
PLOTTER_HEIGHT = 268
PLOTTER_DIAGONAL = 485
drawable_area_offset_left = 15  # left offset
drawable_area_offset_top = -13  # top offset
drawable_area_width = 296  # Width of the drawable area, taking into account the offsets
drawable_area_height = 219  # Height of the drawable area
plotter_drawing_start_offset_x = 0
plotter_drawing_start_offset_y = 0

# new plotters x max is 300
#drawable area 31x21
# new plotters y max is 170
# distance between top left and top right yellow center = 40cm
# distance between top left and bottom left yellow center = 34,6cm
# distance between top left and top right yellow center = 52,4cm

def find_yellow_markers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound_yellow, upper_bound_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

    return centers

def detect_markers_and_calibrate(frame):
    marker_positions = find_yellow_markers(frame)
    calculate_scaling_factors(marker_positions)
    if len(marker_positions) != 3:
        print("Error: Not all markers detected.")
        return None, None, marker_positions
    # Sorting markers by their x-coordinate to align with plotter's coordinate system
    marker_positions.sort(key=lambda x: (x[1], x[0]))
    # Assume markers are placed in the order: bottom-left, top-left, top-right
    top_markers = sorted(marker_positions[:-1], key=lambda x: x[0])  # Sort the top two markers based on x-coordinate
    bottom_marker = marker_positions[-1]  # The last marker is the bottom one
    top_left, top_right = top_markers
    bottom_left = bottom_marker # origin ( near it atleast)
    #sort markers so bottom left becomes origin of plotter
    camera_points = np.array([bottom_left, top_left, top_right], dtype="float32")
    
    plotter_points = np.array([
         [0, 0],  # bottom-left in plotter space
         [0, PLOTTER_HEIGHT],  # top-left in plotter space
         [PLOTTER_WIDTH, PLOTTER_HEIGHT]  # top-right in plotter space
    ], dtype="float32")
   
     # Calculate transformation matrix
    transformation_matrix = cv2.getAffineTransform(camera_points, plotter_points)
    return transformation_matrix, bottom_left, marker_positions

# Function to calculate pixel distance between two points
def calculate_pixel_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Function to calculate scaling factors based on detected markers
def calculate_scaling_factors(marker_positions):
    global scaling_factor
    if len(marker_positions) >= 3:
        # Sort markers by their x-coordinate (assuming top-left, top-right, bottom-left ordering)
        marker_positions.sort(key=lambda x: x[0])
        top_left, top_right, bottom_left = marker_positions[:3]

        # Calculate pixel distances
        pixel_distance_top = calculate_pixel_distance(top_left, top_right)
        pixel_distance_left = calculate_pixel_distance(top_left, bottom_left)
        pixel_distance_diagonal = calculate_pixel_distance(top_right, bottom_left)

        # Calculate scaling factors
        scaling_factor_top = REAL_DISTANCE_TOP / pixel_distance_top
        scaling_factor_left = REAL_DISTANCE_LEFT / pixel_distance_left
        scaling_factor_diagonal = REAL_DISTANCE_DIAGONAL / pixel_distance_diagonal

        scaling_factor = (scaling_factor_diagonal+scaling_factor_left+scaling_factor_top)/3

        return scaling_factor
    else:
        return None


# Function to draw markers and plotter area for visual verification
def visualize_calibration(frame, marker_positions, transformation_matrix):
    # Visualize markers
    for (x, y) in marker_positions:
        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)  # Draw yellow circles at marker positions

    if transformation_matrix is not None:
        inv_transformation_matrix = cv2.invertAffineTransform(transformation_matrix)
        # Define the plotter's origin in plotter space
        plotter_origin = np.array([[0, 0]], dtype="float32")
        # Transform plotter's origin to camera view
        plotter_origin_cam_view = cv2.transform(np.array([plotter_origin]), inv_transformation_matrix)[0].astype(int)
         # Plotter origin in camera view
        cv2.circle(frame, tuple(plotter_origin_cam_view[0]), 10, (0, 0, 255), -1)  # Red circle for plotter's origin
        # Visualize +x and +y directions from the origin
        x_direction = cv2.transform(np.array([[[50, 0]]], dtype="float32"), inv_transformation_matrix)[0][0]
        y_direction = cv2.transform(np.array([[[0, 50]]], dtype="float32"), inv_transformation_matrix)[0][0]
        # Draw arrows for +x and +y axes
        cv2.arrowedLine(frame, tuple(plotter_origin_cam_view[0]), tuple(x_direction.astype(int)), (255, 0, 0), 2, tipLength=0.5)  # +x in blue
        cv2.arrowedLine(frame, tuple(plotter_origin_cam_view[0]), tuple(y_direction.astype(int)), (0, 255, 0), 2, tipLength=0.5)  # +y in green
        
        # Transform each corner of the drawable area back to camera view
        drawable_area_corners = np.array([
            [drawable_area_offset_left, drawable_area_offset_top],  # Top-left
            [drawable_area_offset_left + drawable_area_width, drawable_area_offset_top],  # Top-right
            [drawable_area_offset_left + drawable_area_width, drawable_area_offset_top + drawable_area_height],  # Bottom-right
            [drawable_area_offset_left, drawable_area_offset_top + drawable_area_height]  # Bottom-left
        ], dtype="float32")
        
        drawable_area_corners_cam_view = cv2.transform(np.array([drawable_area_corners]), inv_transformation_matrix)[0].astype(int)
        
        # Draw the drawable area in a different color
        cv2.polylines(frame, [drawable_area_corners_cam_view], isClosed=True, color=(0, 255, 0), thickness=2)


        plotter_corners = np.array([
            [0, 0],
            [PLOTTER_WIDTH, PLOTTER_HEIGHT], 
            [0, PLOTTER_HEIGHT]
        ], dtype="float32")
        plotter_corners_in_camera_view = cv2.transform(np.array([plotter_corners]), inv_transformation_matrix)[0].astype(int)
        # Transform plotter corners to camera view
        plotter_corners_in_camera_view = cv2.transform(np.array([plotter_corners]), inv_transformation_matrix)[0].astype(int)
        cv2.polylines(frame, [plotter_corners_in_camera_view], isClosed=True, color=(255, 0, 0), thickness=2)

    # Display the frame with visual feedback
    cv2.imshow("Calibration Verification", frame)

def calibrate_plotter_camera_system(cap):
    # Capture frame for calibration
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame for calibration")
        return None    
    frame_undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
    marker_positions = find_yellow_markers(frame_undistorted)
    # Check if four markers were detected
    if len(marker_positions) != 3:
        print("Did not detect exactly four markers. Detected:", len(marker_positions))
        return None

    static_markers_camera = np.float32(marker_positions[:3])  # Assuming first three are static
    static_markers_plotter = np.float32([[0, 0], [PLOTTER_WIDTH, 0], [PLOTTER_WIDTH, PLOTTER_HEIGHT]])  # Adjust accordingly
    # Calculate transformation matrix from camera to plotter coordinate system
    transformation_matrix = cv2.getAffineTransform(static_markers_camera, static_markers_plotter)
    
    return transformation_matrix

#G-Code Functions
#g code sender functions
def gcode_sender():
    global commands_waiting
    while True:
        try:
            if commands_waiting < max_buffer_size:
                command = gcode_queue.get_nowait()
            else:
                time.sleep(0.01)  # Briefly pause if we're at or near buffer capacity
                continue
        except queue.Empty:
            continue

        if command == "QUIT":
            break

        print("Sending:", command.strip())
        ser.write(command.encode())
        commands_waiting += 1

        while ser.inWaiting() > 0 or commands_waiting >= max_buffer_size:
            response = ser.readline().decode().strip()
            print("Plotter response:", response)

            if 'ok' in response or 'error' in response:
                commands_waiting -= 1

            

        gcode_queue.task_done()


#helper sending gcode
def send_gcode(command):
    gcode_queue.put(command + "\n")
   
#helper pen up
def pen_up():
    send_gcode("M3 S50")  # Pen up command

def pen_down():
    send_gcode("M3 S0")  # Pen down command for drawing


# Start the GCode sender thread
sender_thread = threading.Thread(target=gcode_sender, daemon=True)
sender_thread.start()   

# Function to generate G-code for the plotter with feedrate needs tuple and transforms
def generate_gcode(x_plot,y_plot, feed_rate=10000):
    # Include the feed rate in each move command
    #safety 
    #x_plot = max(0, min(x_plot, drawable_area_width))
    #y_plot = max(0, min(y_plot, drawable_area_height))
    move_command = f"G01 X{x_plot:.2f} Y{y_plot:.2f} F{feed_rate}"  # Now setting the feed rate here
    return f"{move_command}"

#converter function
def camera_to_plotter(point, transformation_matrix):  # Note the negative offset_y for "up"
    global marker_positions, scaling_factor
    # Convert the point to homogenous coordinates (x, y, 1)
    point_homogenous = np.array([point[0], point[1], 1])
    # Apply the transformation matrix
    transformed_point_homogenous = np.dot(transformation_matrix, point_homogenous)
    # Convert back from homogenous coordinates
    transformed_x, transformed_y = transformed_point_homogenous[:2]
    # offset
    adjusted_x = transformed_x * scaling_factor# - drawable_area_offset_left
    adjusted_y = transformed_y * scaling_factor# - drawable_area_offset_top
    # print(f"Your Scaling Factor is : ", scaling_factor)
    #check in bounds
    x_plot_adjusted = max(0, min(adjusted_x, drawable_area_width))
    y_plot_adjusted = max(0, min(adjusted_y, drawable_area_height))
    print(f"X ={x_plot_adjusted, y_plot_adjusted}")
    return int(x_plot_adjusted), int(y_plot_adjusted)
    

#Voroni Stuff
#helper to plot the diagram

#draw voroni lines
def draw_voronoi(shapes):
    # Ensure points are within plotter bounds before generating Voronoi diagram
    # Assuming 'shapes' is a list of lists of points, where each point is a tuple (x, y)
    bounded_points = [camera_to_plotter(pt, paper_contour) for shape in shapes for pt in shape]
    all_points = np.array(bounded_points)
    
    if len(all_points) < 3:
        print("Not enough points for Voronoi.")
        return
    
    vor = Voronoi(all_points)
    
    # Draw edges ensuring they are within bounds
    for edge in vor.ridge_vertices:
        start, end = edge
        if start >= 0 and end >= 0:
            start_pt = vor.vertices[start]
            end_pt = vor.vertices[end]
            
    
def prepare_gcode_sequence(detected_shapes, paper_contour):
    gcode_sequence = ["M3 S50\n"]  # Pen up at the start of drawing
    for shape in detected_shapes:
        # Process each shape to generate G-code
        for point in shape[1:]:
            transformed_point = camera_to_plotter((point[0][0], point[0][1]), paper_contour)
            # Generate G-code for moving to the point
            gcode_sequence.append(generate_gcode(transformed_point[0], transformed_point[1]))
    gcode_sequence.append("M3 S0\n")  # Pen down after drawing
    return gcode_sequence



# Convert Voronoi vertices directly
def draw_line(start_pt, end_pt):
    x1, y1 = start_pt
    x2, y2 = end_pt
    # Clip to ensure within plotter bounds
    x1, y1 = np.clip(x1, 0, PLOTTER_WIDTH), np.clip(y1, 0, PLOTTER_HEIGHT)
    x2, y2 = np.clip(x2, 0, PLOTTER_WIDTH), np.clip(y2, 0, PLOTTER_HEIGHT)
    send_gcode(generate_gcode(x1, y1))  # Move to start
    pen_down()
    send_gcode(generate_gcode(x2, y2))  # Draw to end
    pen_up()

#detect paper on table
def detect_paper(frame, lower_bound, upper_bound):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        return cv2.approxPolyDP(largest_contour, epsilon, True)
        #return largest_contour
    else:
        return None


def smooth_contour(contour, smoothing_factor=0.002):
    epsilon = smoothing_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def draw_paper_outline(paper_contour):
    if paper_contour is None:
        print("No paper contour detected.")
        return
    flattened_contour = paper_contour.reshape(-1, 2)
    pen_up()

    # Assuming the first point to start
    first_point = flattened_contour[0]
    # Move to the first point without drawing
    x_plot, y_plot = camera_to_plotter(first_point, transformation_matrix)
    send_gcode(f"G0 X{x_plot:.2f} Y{y_plot:.2f}")  # G0 command for rapid movement without drawing
    pen_down()

    # Draw lines to all subsequent points including drawing a line back to the first point to close the loop
    for point in flattened_contour[1:]:
        x_plot, y_plot = camera_to_plotter(point, transformation_matrix)
        send_gcode(generate_gcode(x_plot, y_plot))

    # Explicitly draw line back to the first point to ensure the contour is closed
    x_plot, y_plot = camera_to_plotter(first_point, transformation_matrix)
    send_gcode(generate_gcode(x_plot, y_plot))

    pen_up()

    



#detect shapes on paper within boundaries
def detect_shapes_on_paper(frame, paper_contour):
    global detected_shapes
    detected_shapes.clear()  # Clear previously detected shapes

    if paper_contour is not None:
        # Create a mask from the paper contour
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.drawContours(mask, [paper_contour], -1, (255, 255, 255), -1)  # Fill the contour

        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, mask[:, :, 0])

        # Find contours of shapes in the masked image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Convex Hull of the detected contour
            hull = cv2.convexHull(contour)

            # Smooth the Convex Hull
            smoothed_hull = smooth_contour(hull, smoothing_factor=0.002)

            # Skip small contours that might be noise and overly simplified shapes
            if cv2.contourArea(smoothed_hull) > 100:
                detected_shapes.append(smoothed_hull)



    
def draw_transformed_shape(shape_points_camera_coords):
    # Check if transformation matrix is set
    if transformation_matrix is None:
        print("Transformation matrix not set. Cannot transform shape coordinates.")
        return

    # Transform shape points to plotter coordinates
    shape_points_plotter_coords = [camera_to_plotter(point, transformation_matrix) for point in shape_points_camera_coords]

    # Lower pen for the start of the drawing
    pen_down()

    # Move to the starting point without drawing
    first_point = shape_points_plotter_coords[0]
    send_gcode(f"G0 X{first_point[0]:.2f} Y{first_point[1]:.2f}\n")

    # Lower pen to start drawing
    pen_down()
    # Generate G-code for each point and send to the plotter
    for point in shape_points_plotter_coords[1:]:
        gcode_command = generate_gcode(point[0], point[1])
        send_gcode(gcode_command)

    # Lift pen after finishing drawing
    pen_up()


def handle_mouse_click(event, x, y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(f"Mouse clicked at ({x}, {y})")
        pen_up()
        x_plot, y_plot = camera_to_plotter((x, y), transformation_matrix)
        gcode_command = generate_gcode(x_plot, y_plot)
        send_gcode(gcode_command)
        pen_down()



def on_press(key):
    if key == keyboard.Key.space:
         # Assuming paper_contour is available here; you might need to capture a frame and detect the paper contour if not already done.
        # print("Current Transformation Matrix:", transformation_matrix)
        #draw_transformed_shape(rectangle_points)
        draw_paper_outline(paper_contour)
      
       
        # transformed_shapes = [[camera_to_plotter((pt[0][0], pt[0][1]), paper_contour) for pt in shape] for shape in detected_shapes]
        # flat_points = [pt for shape in transformed_shapes for pt in shape]
        # plot_voronoi_diagram(flat_points)
        # gcode_sequence = prepare_gcode_sequence(detected_shapes, paper_contour)'
        # # if transformed_shapes:
        # #     redraw_shapes(detected_shapes, paper_contour)
        # #     try:
        # #         flat_points = [pt for shape in transformed_shapes for pt in shape]
        # #         plot_voronoi_diagram(flat_points)  # Pass the flattened list to the function
        # #     except ValueError as e:
        # #         print("Error in plotting Voronoi diagram:", e)
            
        # def send_prepared_gcode():
        #     for command in gcode_sequence:
        #         send_gcode(command)
        # threading.Thread(target=send_prepared_gcode).start()    



# Setup the listener
listener = keyboard.Listener(on_press=on_press)
listener.start()
paper_contour = None



def main():
    global paper_contour
    global transformation_matrix
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        cap.release()
    else:
        transformation_matrix, plotter_origin, marker_positions = detect_markers_and_calibrate(frame)
        if transformation_matrix is not None and not np.all((transformation_matrix == 0)):
            
            print("Calibration Successful. Transformation Matrix:", transformation_matrix)
            print("Plotter Origin in Camera Frame:", plotter_origin)
        else:
            print("Calibration Failed.")
            cap.release()
            cv2.destroyAllWindows()
            closeSystem()
            listener.stop()
        cap.release()
    visualize_calibration(frame, marker_positions, transformation_matrix)
    #setup resolution and camera   
    frame_undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
    desired_width = 1280
    desired_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FPS,30)
    cap = cv2.VideoCapture(0)
    print("Camera Online")
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #frame_undistorted = cv2.undistort(frame, mtx, dist, None, mtx)

        # detect paper
        lower_bound = np.array([74, 9, 175])
        upper_bound = np.array([142, 187, 255])
        paper_contour = detect_paper(frame, lower_bound, upper_bound)
        
        if paper_contour is not None:
            detect_shapes_on_paper(frame, paper_contour)
            cv2.drawContours(frame, [paper_contour], -1, (0, 255, 0), 3)
            for shape in detected_shapes:
                cv2.drawContours(frame, [shape], -1, (255, 0, 0), 2)

        cv2.imshow('Master Thesis Vessi', frame)
        cv2.setMouseCallback('Master Thesis Vessi', handle_mouse_click)
  
            
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Master Thesis Vessi', cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    closeSystem()
    listener.stop()
    cv2.destroyAllWindows()
    
#clean function
def closeSystem():
    print("Initiating cleanup...")
    unlock_command = '$X'
    gohome = 'G0 X0 Y0'
    time.sleep(1)
    send_gcode(unlock_command)
    pen_up()
    time.sleep(0.1)
    send_gcode(gohome)  # Move to origin
    time.sleep(3) 
     # Use None as a sentinel value to signal the end
    pen_down()
    time.sleep(0.1)
    ser.close()
    sender_thread.join() 
    print("Camera released and windows closed and Plotter at Origin. Cleanup complete.")
    
if __name__ == "__main__":

    main()
