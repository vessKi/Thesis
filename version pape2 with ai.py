import cv2
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard
import serial  # for arduino
from serial import Serial
import time  # for waiting
import threading 
import queue
from queue import Queue
from shapely.geometry import LineString, Polygon, MultiLineString
import random
import shapely
import tensorflow as tf
import tensorflow_datasets as tfds

#BIG UPDATE
# no longer translating the coordinates within the function just in the gcode generation
# finally have a somewhat accurate translation matrix with a 0,0 and a offset that is not quiete right yet.
# need to adjust every other function to use the new matrix and simplyfy the entire thing by alot


with np.load('calibration_data.npz') as X: # generated with the chess calibration.py and the images in the folder chess
    mtx, dist = [X[i] for i in ('mtx','dist')]

ser = serial.Serial('COM3', 115200, timeout=1)
transformation_matrix = None

# Load the Quick, Draw! dataset
ds_train, ds_test = tfds.load('quickdraw_bitmap', split=['train', 'test'], as_supervised=True)

def preprocess_image(image, label):
    image = tf.image.resize(image, [28, 28]) / 255.0  # Resize and normalize
    return image, label

ds_train = ds_train.map(preprocess_image).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(preprocess_image).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(345, activation='softmax')  # 345 classes in Quick, Draw!
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(ds_train, epochs=10, validation_data=ds_test, callbacks=[early_stopping])

class_names = tfds.builder('quickdraw_bitmap').info.features['label'].names

def predict_drawing(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [28, 28]) / 255.0
    img = tf.expand_dims(img, 0)  # Add batch dimension

    predictions = model.predict(img)
    top_pred = class_names[tf.argmax(predictions[0])]
    return top_pred

image_path = 'path_to_your_image.png'
top_prediction = predict_drawing(image_path)
print(f"Top prediction: {top_prediction}")

def get_drawing_suggestions(label):
    suggestions = {
        'cat': 'Draw a ball of yarn.',
        'dog': 'Draw a bone.',
        'flower': 'Draw a butterfly.'
    }
    return suggestions.get(label, 'No suggestion available')

suggestion = get_drawing_suggestions(top_prediction)
print(f"Suggestion: {suggestion}")

# Store calibration points
# Global variables for calibration
expected_positions = []
actual_positions = []
calibration_phase = 0  # 0 for setting expected point, 1 for setting actual point


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
lower_bound_yellow = np.array([12, 132, 167])
upper_bound_yellow = np.array([67, 187, 255])
# in mm
REAL_DISTANCE_TOP = 390
REAL_DISTANCE_LEFT = 270
REAL_DISTANCE_DIAGONAL = 485
#in mm
PLOTTER_WIDTH = 390
PLOTTER_HEIGHT = 270
PLOTTER_DIAGONAL = 485
drawable_area_offset_left = 17 # left offset
drawable_area_offset_top = -5  # top offset
drawable_area_width = 300  # Width of the drawable area, taking into account the offsets
drawable_area_height = 220  # Height of the drawable area


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
            command = gcode_queue.get(timeout=1)  # Wait for a command from the queue
            if command == "QUIT":
                break
            while commands_waiting >= max_buffer_size:
                time.sleep(0.1)  # Delay before checking again to reduce CPU usage

            ser.write(command.encode())
            commands_waiting += 1
            print(f"Sending: {command.strip()}")

            # Wait for an acknowledgment from the plotter before sending the next command
            while commands_waiting > 0:
                response = ser.readline().decode().strip()
                print("Plotter response:", response)
                if 'ok' in response or 'error' in response:
                    commands_waiting -= 1  # Decrease count only when acknowledged
        except serial.SerialException as e:
            print(f"Serial communication error: {e}")
        except queue.Empty:
        # Handle case where the queue is empty and we timeout
            pass
       

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

#i hate this function
def camera_to_plotter(point, transformation_matrix):
    global scaling_factor
    # Convert the point to homogenous coordinates (x, y, 1)
    point_homogenous = np.array([point[0], point[1], 1]).reshape((3, 1))
    # Apply the transformation matrix
    transformed_point_homogenous = np.dot(transformation_matrix, point_homogenous).flatten()
    # Convert back from homogenous coordinates
    transformed_x, transformed_y = transformed_point_homogenous[:2]
    # Apply scaling factor
    adjusted_x = transformed_x * scaling_factor
    adjusted_y = transformed_y * scaling_factor
    # Log the transformed coordinates
    print(f"Original: {point}, Transformed: ({adjusted_x}, {adjusted_y})")
    # Ensure the coordinates are within the drawable area
    x_plot_adjusted = max(0, min(adjusted_x, drawable_area_width))
    y_plot_adjusted = max(0, min(adjusted_y, drawable_area_height))
    return int(x_plot_adjusted), int(y_plot_adjusted)



def clip_line_to_contour(start_point, end_point, contour_polygon):
    line = LineString([start_point, end_point])
    clipped_line = line.intersection(contour_polygon)
    if not clipped_line.is_empty:
        # Check if the intersection result is a LineString
        if isinstance(clipped_line, LineString):
            return [np.array(clipped_line.coords)]
        # Check if the intersection result is a MultiLineString
        elif isinstance(clipped_line, MultiLineString):
            return [np.array(line.coords) for line in clipped_line.geoms]
    return None




def draw_line(start_pt, end_pt):
    # Send G-code to move to the start point
    send_gcode(generate_gcode(start_pt[0], start_pt[1]))  # Move to start
    pen_down()
    # Draw line to the end point
    send_gcode(generate_gcode(end_pt[0], end_pt[1]))
    pen_up()


#detect paper 
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
    arc_length = cv2.arcLength(contour, True)
    epsilon = smoothing_factor * arc_length
    if arc_length == 0:  # This is a critical check if contour is trivial or has issues
        return contour
    return cv2.approxPolyDP(contour, epsilon, True)


    
def prepare_gcode_sequence(detected_shapes, transformation_matrix):
    gcode_sequence = ["M3 S50\n"]
    for shape in detected_shapes:
        for point in shape[1:]:
            transformed_point = camera_to_plotter((point[0][0], point[0][1]), transformation_matrix)
            gcode_sequence.append(generate_gcode(transformed_point[0], transformed_point[1]))
    gcode_sequence.append("M3 S0\n")
    gcode_sequence.append("G0 X0 Y0\n")  # Move back to origin
    return gcode_sequence


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









def on_press(key):
    global detected_shapes, transformation_matrix, paper_contour
    if key == keyboard.Key.space:
        if transformation_matrix is not None and paper_contour is not None:
            # Shuffle shapes before drawing
            random.shuffle(detected_shapes)


            # Generate G-code for drawing shapes
            gcode_commands = prepare_gcode_sequence(detected_shapes, transformation_matrix)
            print("Review the following G-code commands:")
            for command in gcode_commands:
                print(command)

            if input("Send G-code to plotter? (y/n): ").lower() == 'y':
                for command in gcode_commands:
                    send_gcode(command)
                print("G-code sent to plotter.")
            else:
                print("G-code sending aborted.")



# Setup the listener
listener = keyboard.Listener(on_press=on_press)
listener.start()
paper_contour = None

#utlra important safety feature, didnt know id need this 
def collect_gcode_commands(vor, paper_contour):
    gcode_commands = []
    gcode_commands.append("M3 S50")  # Pen up after drawing each line
    if vor is not None:
        contour_polygon = Polygon(paper_contour.reshape(-1, 2))
        for ridge_points in vor.ridge_vertices:
            if all(v >= 0 for v in ridge_points):
                start_point = tuple(vor.vertices[ridge_points[0]])
                end_point = tuple(vor.vertices[ridge_points[1]])
                clipped_line = clip_line_to_contour(start_point, end_point, contour_polygon)
                if clipped_line is not None:
                    for start, end in clipped_line:
                        start_plotter = camera_to_plotter(start, transformation_matrix)
                        end_plotter = camera_to_plotter(end, transformation_matrix)
                        gcode_commands.append("M3 S0")  # Pen down
                        gcode_commands.append(generate_gcode(start_plotter[0], start_plotter[1]))
                        gcode_commands.append(generate_gcode(end_plotter[0], end_plotter[1]))
                        gcode_commands.append("M3 S50")  # Pen up after drawing each line
    gcode_commands.append("G0 X0 Y0")  # Move to origin at the end of drawing
    return gcode_commands


def main():
    global paper_contour
    global transformation_matrix
    global adjustments_made
    global need_calibration_confirmation
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
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Camera Online")
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        

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
