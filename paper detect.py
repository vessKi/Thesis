import cv2
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard
import serial  # for arduino
import time  # for waiting
import threading 
import queue
from queue import Queue
import random # for shape selection
from scipy.spatial import Voronoi, voronoi_plot_2d

# Open serial connection to Plotter
ser = serial.Serial('COM4', 115200) 
PLOTTER_WIDTH = 300
PLOTTER_HEIGHT = 220
gcode_queue = queue.Queue() # for gcode send
main_thread_tasks = queue.Queue() # for plots

if not ser.isOpen():
    ser.open()
soft_reset_command = b'\x18'  # GRBL soft-reset command
ser.write(soft_reset_command)
time.sleep(0.1)
ser.flushInput()  # Clear any data from the input buffer
ser.flushOutput()  # Clear any data from the output buffer
print("Plotter Awake and Listening")
time.sleep(0.1)
ser.write("M3 S50\n".encode())

# Global variable to store detected shapes
detected_shapes = []
commands_waiting = 0 



#GCode Functions

# Function to generate G-code for the plotter with feedrate
def generate_gcode(x_plot, y_plot, feed_rate=10000):
    # Include the feed rate in each move command
    move_command = f"G01 X{x_plot:.2f} Y{y_plot:.2f} F{feed_rate}"  # Now setting the feed rate here
    return f"{move_command}\n"

#g code sender function
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

#helper sending gcode
def send_gcode(command):
    gcode_queue.put(command)
   
#helper pen up
def pen_up():
    send_gcode("M3 S50\n")  # Pen up command
    print("Pen lifted")
    
#helper pen down
def pen_down():
    send_gcode("M3 S25\n")  # Pen down command for drawing
    print("Pen lowered")

# Start the GCode sender thread
sender_thread = threading.Thread(target=gcode_sender, daemon=True)
sender_thread.start()   

#test function 
def redraw_shapes(detected_shapes, paper_contour):
    for shape in detected_shapes:
        # Ensure the pen is up before moving to the start of a new shape
        pen_up()
        
        # Transform the first point and move the plotter pen to the start position without drawing
        start_point = shape[0]  # First point of the shape
        transformed_start = camera_to_plotter((start_point[0][0], start_point[0][1]), paper_contour)
        send_gcode(f"G0 X{transformed_start[0]:.2f} Y{transformed_start[1]:.2f}")  # Move to start
        
        # Lower the pen to start drawing
        pen_down()
        
        # Iterate over the remaining points in the shape, transforming and drawing each segment
        for point in shape[1:]:
            transformed_point = camera_to_plotter((point[0][0], point[0][1]), paper_contour)
            send_gcode(generate_gcode(transformed_point[0], transformed_point[1]))
        
        # Lift the pen after finishing the shape
        pen_up()


#converter function
def camera_to_plotter(point, paper_contour, cam_resolution=(1280, 720), plotter_area=(PLOTTER_WIDTH, PLOTTER_HEIGHT)):
    # Calculate paper size and position in camera coordinates
    x, y, w, h = cv2.boundingRect(paper_contour)
    
    # Camera to plotter scale factors
    scale_x = plotter_area[0] / cam_resolution[0]
    scale_y = plotter_area[1] / cam_resolution[1]

    # Adjust point's position based on paper contour in camera view
    # The X position of the point in plotter coordinates needs to consider the inversion and scaling
    transformed_x = (cam_resolution[0] - (point[0] - x)) * scale_x
    
    # The Y position in plotter coordinates is straightforward scaled
    transformed_y = (point[1] - y) * scale_y

    # Ensure the point is within plotter boundaries
    transformed_x = max(0, min(plotter_area[0], transformed_x))
    transformed_y = max(0, min(plotter_area[1], transformed_y))
    
    return transformed_x, transformed_y

#Voroni Stuff
#helper to plot the diagram
def plot_voronoi_diagram(points):
    def plot():
        print("Preparing to generate Voronoi diagram with points:", points)
        vor = Voronoi(points)
        # Plot using Matplotlib
        fig, ax = plt.subplots()
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='green', line_width=2, line_alpha=0.6, point_size=2)
        ax.set_xlim([0, 1280])  # Adjust the x-axis limits
        ax.set_ylim([0, 720])  # Adjust the y-axis limits
        ax.set_title("Voronoi Diagram")
        plt.show()
    main_thread_tasks.put(plot)
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
            if is_point_within_bounds(start_pt) and is_point_within_bounds(end_pt):
                draw_line(start_pt, end_pt)
    
def prepare_gcode_sequence(detected_shapes, paper_contour):
    gcode_sequence = ["M3 S50\n"]  # Pen up at the start of drawing
    for shape in detected_shapes:
        # Process each shape to generate G-code
        for point in shape[1:]:
            transformed_point = camera_to_plotter((point[0][0], point[0][1]), paper_contour)
            # Generate G-code for moving to the point
            gcode_sequence.append(generate_gcode(transformed_point[0], transformed_point[1]))
    gcode_sequence.append("M3 S30\n")  # Pen down after drawing
    return gcode_sequence

#helper function
def is_point_within_bounds(point):
    x, y = point
    return 0 <= x <= PLOTTER_WIDTH and 0 <= y <= PLOTTER_HEIGHT

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
        return largest_contour
    else:
        return None

#possible fix to shit shapes Deuglas Peucker algo
def smooth_contour(contour, smoothing_factor=0.002):
    epsilon = smoothing_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def draw_paper_outline(paper_contour):
    if paper_contour is None:
        print("No paper contour detected.")
        return

    # Assuming the paper contour can be approximated as a rectangle,
    # we find the bounding rect of the detected contour which gives us the corners
    rect = cv2.minAreaRect(paper_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Ensure the pen is lifted before moving to the first corner
    pen_up()

    # Move to the first corner without drawing
    first_corner = camera_to_plotter(box[0], paper_contour)
    send_gcode(f"G0 X{first_corner[0]:.2f} Y{first_corner[1]:.2f}")

    # Lower the pen to start drawing
    pen_down()

    # Draw lines connecting each corner
    for i in range(1, len(box)):
        corner = camera_to_plotter(box[i], paper_contour)
        send_gcode(generate_gcode(corner[0], corner[1]))

    # Close the rectangle by drawing a line back to the first corner
    send_gcode(generate_gcode(first_corner[0], first_corner[1]))

    # Lift the pen after drawing the outline
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
#plot to check if all good
def plot_paper_and_shapes(paper_contour, shapes):
    def plot():
        if paper_contour is not None:
            x, y, w, h = cv2.boundingRect(paper_contour)
            plt.plot([x, x+w], [y, y], 'r-')  # Top edge
            plt.plot([x, x], [y, y+h], 'r-')  # Left edge
            plt.plot([x+w, x+w], [y, y+h], 'r-')  # Right edge
            plt.plot([x, x+w], [y+h, y+h], 'r-')  # Bottom edge
        for shape in shapes:
            shape = shape.squeeze()
            plt.plot(np.append(shape[:, 0], shape[0, 0]), np.append(shape[:, 1], shape[0, 1]), 'g-')
        plt.xlim([0, 1280])
        plt.ylim([0, 720])
        plt.title("Detected Paper and Shapes")
        plt.show()
    main_thread_tasks.put(plot)
    
# Assuming these are the points on the paper as seen by the camera
# These should be replaced by your actual measurements or estimates
camera_points = [(109, 115), (597, 117), (597, 468), (109, 468)]  # Example points

def camera_to_plotter_test(point, cam_resolution=(1280, 720), plotter_area=(PLOTTER_WIDTH, PLOTTER_HEIGHT)):
    """
    Adjust this function based on your observations of how the camera's coordinates
    map to the plotter's coordinates.
    """
    # Corrected scale factors based on physical measurements
    scale_x = plotter_area[0] / cam_resolution[0]
    scale_y = plotter_area[1] / cam_resolution[1]

    # Apply transformations considering origin differences and axis orientations
    # Assuming the need to invert Y axis and adjust for a different origin location
    transformed_x = (cam_resolution[0] - point[0]) * scale_x  # Invert X axis if necessary
    transformed_y = point[1] * scale_y  # Invert Y axis if necessary
    
    # Ensure the points are within plotter boundaries
    transformed_x = max(0, min(plotter_area[0], transformed_x))
    transformed_y = max(0, min(plotter_area[1], transformed_y))

    return transformed_x, transformed_y


def draw_test_points():
    """
    Draws lines between test points to help calibrate the coordinate transformation.
    """
    pen_up()
    
    # Move to the first point without drawing
    first_point = camera_to_plotter_test(camera_points[0])
    send_gcode(f"G0 X{first_point[0]:.2f} Y{first_point[1]:.2f}")
    
    pen_down()
    
    # Draw lines to the rest of the points
    for point in camera_points[1:]:
        plotter_point = camera_to_plotter_test(point)
        send_gcode(generate_gcode(plotter_point[0], plotter_point[1]))
    
    # Close the loop by drawing a line back to the first point
    send_gcode(generate_gcode(first_point[0], first_point[1]))
    
    pen_up()    
#on space plot and then draw
def on_press(key):
    if key == keyboard.Key.space:
        draw_test_points()
        #draw_paper_outline(paper_contour)
        #plot_paper_and_shapes(paper_contour, detected_shapes) # plot paper
        transformed_shapes = [[camera_to_plotter((pt[0][0], pt[0][1]), paper_contour) for pt in shape] for shape in detected_shapes]
        flat_points = [pt for shape in transformed_shapes for pt in shape]
        plot_voronoi_diagram(flat_points)
        gcode_sequence = prepare_gcode_sequence(detected_shapes, paper_contour)
        # # if transformed_shapes:
        # #     redraw_shapes(detected_shapes, paper_contour)
        # #     try:
        # #         flat_points = [pt for shape in transformed_shapes for pt in shape]
        # #         plot_voronoi_diagram(flat_points)  # Pass the flattened list to the function
        # #     except ValueError as e:
        # #         print("Error in plotting Voronoi diagram:", e)
            
        def send_prepared_gcode():
            for command in gcode_sequence:
                send_gcode(command)
        threading.Thread(target=send_prepared_gcode).start()    


paper_contour = None
# Setup the listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Start the plotting thread
def main():
    global paper_contour
    
    
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    desired_width = 1280
    desired_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    print("Camera Online")
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # detect paper
        lower_bound = np.array([63, 0, 40])
        upper_bound = np.array([158, 156, 215])
        paper_contour = detect_paper(frame, lower_bound, upper_bound)
        
        if paper_contour is not None:
            detect_shapes_on_paper(frame, paper_contour)
            cv2.drawContours(frame, [paper_contour], -1, (0, 255, 0), 3)
            for shape in detected_shapes:
                cv2.drawContours(frame, [shape], -1, (255, 0, 0), 2)

        cv2.imshow('Master Thesis Vessi', frame)
        #threading logic
        while not main_thread_tasks.empty():
            task = main_thread_tasks.get()
            task()
            
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Master Thesis Vessi', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    closeSystem()
    listener.stop()
    
#clean function
def closeSystem():
    print("Initiating cleanup...")
    soft_reset_command = b'\x18'  # GRBL soft-reset command
    unlock_command = b'$X\n'
    time.sleep(1)  # Wait for the reset to process
    ser.write(soft_reset_command)
    time.sleep(1)
    ser.write(unlock_command)
    time.sleep(1) 
    ser.flushInput()  # Clear any data in the input buffer
    ser.flushOutput()  # Clear any data in the output buffer
    time.sleep(0.2)
    pen_up()
    time.sleep(0.1)
    ser.write(b'G90\n')  # Ensure absolute positioning mode
    ser.write(b'G0 X0 Y0\n')  # Move to origin
    time.sleep(5) 
    pen_down()
    time.sleep(0.1)
    ser.close()
    gcode_queue.put("QUIT")
    sender_thread.join() 
    print("Camera released and windows closed and Plotter at Origin. Cleanup complete.")

if __name__ == "__main__":
    main()
