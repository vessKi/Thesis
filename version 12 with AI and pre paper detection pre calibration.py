import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import math
import random
import tensorflow_datasets as tfds
import os
import serial
import time
import threading
import queue
import json
from pynput import keyboard
from queue import Queue
import requests
import base64
import ndjson
import svgwrite
from svgpathtools import svg2paths2
from datetime import datetime
import svgpathtools

# Load calibration data
with np.load('c:/Users/kilia/Desktop/COOP/Thesis/calibration_data.npz') as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

ser = serial.Serial('COM4', 115200, timeout=1)
transformation_matrix = None
current_pen_position = (0, 0)  # Initial pen position
JOG_STEP_SIZE = 1  # Define the step size for jogging in units

# Initialize the pen's position and other variables
pen_is_moving = False

# Load the categories
def load_categories(file_path='categories.txt'):
    with open(file_path, 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    return categories

categories = load_categories()
model_path = 'c:/Users/kilia/Desktop/COOP/Thesis/quickdraw_model.keras'

# Load the pre-trained model
print("Loading existing model...")
model = tf.keras.models.load_model(model_path)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# for predefinied drawings
svg_output_dir = r'c:/Users/kilia/Desktop/COOP/Thesis/pre_svg'
os.makedirs(svg_output_dir, exist_ok=True)

def preprocess_image(image):
    resized = cv2.resize(image, (28, 28))  # Resize to the required shape
    normalized = resized / 255.0           # Normalize to range [0, 1]
    
    # Flatten the image and reshape it to match the sequence length and feature dimensions
    flattened = normalized.flatten()
    
    # Pad or truncate the flattened data to ensure it matches the required sequence length
    sequence_length = 1309
    if len(flattened) > sequence_length * 3:
        flattened = flattened[:sequence_length * 3]
    else:
        flattened = np.pad(flattened, (0, sequence_length * 3 - len(flattened)), 'constant')
    
    reshaped = flattened.reshape((1, sequence_length, 3))
    
    return reshaped

def predict_drawing(thresh):
    resized = cv2.resize(thresh, (28, 28))  # Resize to the required shape
    normalized = resized / 255.0            # Normalize to range [0, 1]
    
    # Flatten the image and reshape it to match the sequence length and feature dimensions
    flattened = normalized.flatten()
    
    # Pad or truncate the flattened data to ensure it matches the required sequence length
    sequence_length = 1309
    if len(flattened) > sequence_length * 3:
        flattened = flattened[:sequence_length * 3]
    else:
        flattened = np.pad(flattened, (0, sequence_length * 3 - len(flattened)), 'constant')
    
    reshaped = flattened.reshape((1, sequence_length, 3))
    print(f"Preprocessed image shape: {reshaped.shape}")  # Debugging print

    predictions = model.predict(reshaped)
    top_prediction_index = np.argmax(predictions[0])
    top_prediction_label = categories[top_prediction_index]
    top_prediction_probability = predictions[0][top_prediction_index]
    save_prediction(top_prediction_label, top_prediction_probability, thresh)
    return top_prediction_index, top_prediction_label, top_prediction_probability

def save_prediction(prediction, probability, thresh_image):
    # Create the directory if it doesn't exist
    prediction_dir = "c:/Users/kilia/Desktop/COOP/Thesis/prediction_images"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    # Create the timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the threshold image as a PNG file
    png_filename = os.path.join(prediction_dir, f"prediction_{prediction}_{timestamp}.png")
    cv2.imwrite(png_filename, thresh_image)

    # Create the SVG filename
    svg_filename = os.path.join(prediction_dir, f"prediction_{prediction}_{timestamp}.svg")

    # Create the SVG drawing
    dwg = svgwrite.Drawing(svg_filename, profile='tiny')

    # Add text with the prediction and probability
    dwg.add(dwg.text(f'Prediction: {prediction}', insert=(10, 20), fill='black'))
    dwg.add(dwg.text(f'Probability: {probability:.2f}', insert=(10, 40), fill='black'))

    # Embed the PNG image into the SVG
    dwg.add(dwg.image(href=png_filename, insert=(10, 60), size=("256px", "256px")))

    # Save the SVG file
    dwg.save()




def create_trackbar():
    cv2.namedWindow('Thresholded Frame')
    cv2.createTrackbar('Threshold', 'Thresholded Frame', 150, 255, on_trackbar)

def on_trackbar(val):
    global current_frame, saved_corners
    if current_frame is not None and saved_corners is not None:
        detect_shapes_on_paper(current_frame.copy(), saved_corners, threshold_val=val)

# Your other code for the drawing application

# Store calibration points
# Global variables for calibration
gcode_queue = queue.Queue()  # for G-code send
global scaling_factor
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

# in mm
PLOTTER_WIDTH = 297  # A4 paper width in mm
PLOTTER_HEIGHT = 210  # A4 paper height in mm
MARGIN = 50

# File to save/load corner coordinates
corners_file = 'corners.json'

def load_corners():
    if os.path.exists(corners_file):
        with open(corners_file, 'r') as f:
            return json.load(f)
    return None

def copy_detected_shapes():
    if transformation_matrix is not None and detected_shapes:
        pen_up()
        for shape in detected_shapes:
            if len(shape) == 0:
                continue  # Skip empty shapes

            # Move to the starting point of the shape without drawing
            start_point = shape[0][0]  # Get the first point of the shape
            start_plotter_coords = camera_to_plotter((start_point[0], start_point[1]), transformation_matrix)
            if not is_within_plotter_margins(start_plotter_coords):
                continue  # Skip if the start point is outside the margins

            send_gcode(generate_gcode(start_plotter_coords[0], start_plotter_coords[1]))

            # Draw the shape
            pen_down()
            for point in shape:
                plotter_coords = camera_to_plotter((point[0][0], point[0][1]), transformation_matrix)
                if is_within_plotter_margins(plotter_coords):
                    send_gcode(generate_gcode(plotter_coords[0], plotter_coords[1]))

            # Ensure the shape is closed by drawing a line back to the start
            send_gcode(generate_gcode(start_plotter_coords[0], start_plotter_coords[1]))
        pen_up()
        send_gcode("G0 X0 Y0")  # Move back to origin
        
def detect_lines(frame, threshold_val=150):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours


def draw_lines(shapes):
    max_x = PLOTTER_WIDTH - 2 * MARGIN
    max_y = PLOTTER_HEIGHT - 2 * MARGIN
    random_start_x = random.uniform(MARGIN, max_x)
    random_start_y = random.uniform(MARGIN, max_y)

    adjusted_gcode = []
    current_position = None

    for shape in shapes:
        if len(shape) == 0:
            continue

        start_point = shape[0][0]
        new_start_x = start_point[0] + random_start_x
        new_start_y = start_point[1] + random_start_y

        if not is_within_plotter_margins((new_start_x, new_start_y)):
            continue

        if current_position != (new_start_x, new_start_y):
            adjusted_gcode.append(generate_gcode(new_start_x, new_start_y))
            current_position = (new_start_x, new_start_y)
        adjusted_gcode.append("M3 S0")  # Pen down command for drawing

        for point in shape:
            new_x = point[0][0] + random_start_x
            new_y = point[0][1] + random_start_y

            if is_within_plotter_margins((new_x, new_y)) and current_position != (new_x, new_y):
                adjusted_gcode.append(generate_gcode(new_x, new_y))
                current_position = (new_x, new_y)

        adjusted_gcode.append("M3 S50")  # Pen up command after drawing

    adjusted_gcode.append("G0 X0 Y0")  # Move back to origin

    # Send G-code commands in one batch
    for line in adjusted_gcode:
        send_gcode(line.strip())

    print(f"G-code for detected lines sent to the robot at random position ({random_start_x:.2f}, {random_start_y:.2f}).")

def is_within_plotter_margins(coords, margin=MARGIN):
    x, y = coords
    return margin <= x <= (PLOTTER_WIDTH - margin) and margin <= y <= (PLOTTER_HEIGHT - margin)

def optimize_lines(contours):
    optimized_shapes = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)  # Use a moderate epsilon value to reduce points
        approx = cv2.approxPolyDP(contour, epsilon, True)
        optimized_shapes.append(approx)

    return optimized_shapes



def apply_transformation_matrix(frame, corners):
    if corners:
        # Define plotter points corresponding to A4 paper with origin at top-right
        plotter_points = np.array([
            [0, 0],  # top-right in plotter space
            [0, PLOTTER_HEIGHT],  # bottom-right in plotter space
            [PLOTTER_WIDTH, PLOTTER_HEIGHT],  # bottom-left in plotter space
            [PLOTTER_WIDTH, 0]  # top-left in plotter space
        ], dtype="float32")
        
        # Convert corners to float32
        corners = np.array(corners, dtype="float32")
        
        # Calculate transformation matrix
        transformation_matrix = cv2.getPerspectiveTransform(corners, plotter_points)
        return transformation_matrix
    return None

def visualize_calibration(frame, corners, transformation_matrix):
    # Visualize paper corners
    if corners:
        for point in corners:
            cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)
    
    if transformation_matrix is not None:
        inv_transformation_matrix = np.linalg.inv(transformation_matrix)
        # Define the plotter's origin in plotter space
        plotter_origin = np.array([[PLOTTER_WIDTH, 0]], dtype="float32")
        # Transform plotter's origin to camera view
        plotter_origin_cam_view = cv2.perspectiveTransform(np.array([plotter_origin]), inv_transformation_matrix)[0].astype(int)
        # Plotter origin in camera view
        cv2.circle(frame, tuple(plotter_origin_cam_view[0]), 10, (0, 0, 255), -1)  # Red circle for plotter's origin
        # Visualize +x and +y directions from the origin
        x_direction = cv2.perspectiveTransform(np.array([[[PLOTTER_WIDTH - 50, 0]]], dtype="float32"), inv_transformation_matrix)[0][0]
        y_direction = cv2.perspectiveTransform(np.array([[[PLOTTER_WIDTH, 50]]], dtype="float32"), inv_transformation_matrix)[0][0]
        # Draw arrows for +x and +y axes
        cv2.arrowedLine(frame, tuple(plotter_origin_cam_view[0]), tuple(x_direction.astype(int)), (255, 0, 0), 2, tipLength=0.5)  # +x in blue
        cv2.arrowedLine(frame, tuple(plotter_origin_cam_view[0]), tuple(y_direction.astype(int)), (0, 255, 0), 2, tipLength=0.5)  # +y in green

    # Display the frame with visual feedback
    cv2.imshow("Calibration Verification", frame)

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
                try:
                    response = ser.readline().decode().strip()
                    print("Plotter response:", response)
                    if 'ok' in response or 'error' in response:
                        commands_waiting -= 1  # Decrease count only when acknowledged
                except serial.SerialException as e:
                    print(f"Serial read error: {e}")
                    break
        except serial.SerialException as e:
            print(f"Serial communication error: {e}")
        except queue.Empty:
            pass

def send_gcode(command):
    try:
        gcode_queue.put(command + "\n")
    except Exception as e:
        print(f"Failed to put command in queue: {e}")

def pen_up():
    send_gcode("M3 S50")  # Pen up command

def pen_down():
    send_gcode("M3 S0")  # Pen down command for drawing



# Start the G-Code sender thread
sender_thread = threading.Thread(target=gcode_sender, daemon=True)
sender_thread.start()

def generate_gcode(x_plot, y_plot, feed_rate=8000):
    move_command = f"G01 X{x_plot:.2f} Y{y_plot:.2f} F{feed_rate}"
    return f"{move_command}"

def draw_paper_border():
    if transformation_matrix is not None:
        # Define the corners of the paper in order starting from top-right and moving clockwise
        corners = np.array([
            [0, 0],  # top-right
            [0, PLOTTER_HEIGHT],  # bottom-right
            [PLOTTER_WIDTH, PLOTTER_HEIGHT],  # bottom-left
            [PLOTTER_WIDTH, 0]  # top-left
        ])

        # Transform the corners to plotter coordinates
        plotter_corners = [camera_to_plotter(corner, transformation_matrix) for corner in corners]

        # Debugging: Print the plotter coordinates to verify correctness
        print("Plotter Corners:")
        for i, corner in enumerate(plotter_corners):
            print(f"Corner {i}: {corner}")

        # Draw the border by moving the plotter to each corner
        pen_up()
        send_gcode(generate_gcode(plotter_corners[0][0], plotter_corners[0][1]))  # Move to the starting corner
        pen_down()
        for corner in plotter_corners[1:]:
            send_gcode(generate_gcode(corner[0], corner[1]))
        send_gcode(generate_gcode(plotter_corners[0][0], plotter_corners[0][1]))  # Close the loop
        pen_up()


def crop_to_paper(frame, corners):
    # Convert corners to float32
    corners = np.array(corners, dtype="float32")

    # Define the destination points (A4 paper size in pixels)
    width, height = 297, 210  # A4 size in mm
    dst_points = np.array([
        [0, 0],  # top-right
        [0, height],  # bottom-right
        [width, height],  # bottom-left
        [width, 0]  # top-left
    ], dtype="float32")

    # Calculate the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)

    # Perform the perspective transformation
    cropped_image = cv2.warpPerspective(frame, transform_matrix, (width, height))

    return cropped_image

# def draw_result(category):

#     gcode_dir = 'c:/Users/kilia/Desktop/COOP/Thesis/gcode'
#     gcode_file = os.path.join(gcode_dir, f'{category}.gcode')

#     if os.path.exists(gcode_file):
#         with open(gcode_file, 'r') as file:
#             gcode_lines = file.readlines()

#         for line in gcode_lines:
#             send_gcode(line.strip())
#         print(f"G-code for category '{category}' sent to the robot.")
#     else:
#         print(f"No G-code file found for category '{category}'.")

def draw_result(category):
    data_dir = r'C:\Users\kilia\Desktop\COOP\quickdraw_dataset'
    svg_output_dir = r'C:\Users\kilia\Desktop\COOP\Thesis\output'
    svg_file = extract_svg_from_ndjson(data_dir, svg_output_dir, category)

    if svg_file:
        gcode = svg_to_gcode(svg_file, scale=0.2, feed_rate=8000)
        
        # Calculate random starting position within bounds
        max_x = PLOTTER_WIDTH - 2 * MARGIN
        max_y = PLOTTER_HEIGHT - 2 * MARGIN
        random_start_x = random.uniform(MARGIN/2, max_x)
        random_start_y = random.uniform(MARGIN/2, max_y)
        
        # Adjust G-code to start from the random position
        adjusted_gcode = []
        for line in gcode:
            if line.startswith("G0") or line.startswith("G1"):
                parts = line.split()
                x_val = None
                y_val = None
                for part in parts:
                    if part.startswith("X"):
                        x_val = float(part[1:])
                    elif part.startswith("Y"):
                        y_val = float(part[1:])
                
                if x_val is not None and y_val is not None:
                    new_x = x_val + random_start_x
                    new_y = y_val + random_start_y
                    adjusted_gcode.append(f"{parts[0]} X{new_x:.2f} Y{new_y:.2f} {' '.join(parts[3:])}")
                else:
                    adjusted_gcode.append(line)
            else:
                adjusted_gcode.append(line)
        adjusted_gcode.append("M3 S50")  # Lift the pen
        adjusted_gcode.append("G0 X0 Y0")  # Move back to origin
        for line in adjusted_gcode:
            send_gcode(line.strip())
        print(f"G-code for category '{category}' sent to the robot at random position ({random_start_x:.2f}, {random_start_y:.2f}).")
        adjusted_gcode.append("G0 X0 Y0")
    else:
        print(f"No SVG file found for category '{category}'.")


def extract_svg_from_ndjson(data_dir, output_dir, category):
    os.makedirs(output_dir, exist_ok=True)
    file = f'{category}.ndjson'
    file_path = os.path.join(data_dir, file)
    
    if os.path.exists(file_path):
        with open(file_path) as f:
            data = ndjson.loads(f.read())
            if data:
                drawing = random.choice(data)['drawing']
                svg_path = os.path.join(output_dir, f'{category}.svg')
                dwg = svgwrite.Drawing(svg_path, size=(256, 256))
                for stroke in drawing:
                    points = list(zip(stroke[0], stroke[1]))
                    dwg.add(dwg.polyline(points, stroke=svgwrite.rgb(0, 0, 0, '%'), fill='none', stroke_width=2))
                dwg.save()
                return svg_path
    return None

def svg_to_gcode(svg_file, scale=1.0, feed_rate=10000):
    paths, attributes, svg_attributes = svg2paths2(svg_file)
    gcode = []

    for path in paths:
        gcode.append("M3 S50")  # Lift the pen
        start_point = path[0].start
        gcode.append(f"G0 X{start_point.real * scale:.2f} Y{start_point.imag * scale:.2f}")  # Move to the start of the path
        gcode.append("M3 S0")  # Lower the pen

        for segment in path:
            if isinstance(segment, svgpathtools.Line):
                end = segment.end
                gcode.append(f"G1 X{end.real * scale:.2f} Y{end.imag * scale:.2f} F{feed_rate}")
            elif isinstance(segment, svgpathtools.QuadraticBezier):
                num_points = 10
                points = [segment.point(t / num_points) for t in range(num_points + 1)]
                for point in points:
                    gcode.append(f"G1 X{point.real * scale:.2f} Y{point.imag * scale:.2f} F{feed_rate}")
            elif isinstance(segment, svgpathtools.CubicBezier):
                num_points = 10
                points = [segment.point(t / num_points) for t in range(num_points + 1)]
                for point in points:
                    gcode.append(f"G1 X{point.real * scale:.2f} Y{point.imag * scale:.2f} F{feed_rate}")
            elif isinstance(segment, svgpathtools.Arc):
                num_points = 10
                points = [segment.point(t / num_points) for t in range(num_points + 1)]
                for point in points:
                    gcode.append(f"G1 X{point.real * scale:.2f} Y{point.imag * scale:.2f} F{feed_rate}")

    gcode.append("M3 S50")  # Lift the pen at the end
    gcode.append("G0 X0 Y0") #move back to origin
    return gcode







def create_l_system(dwg, axiom, rules, angle, length, iterations, start_position, start_angle):
    def apply_rules(axiom):
        return "".join(rules.get(char, char) for char in axiom)

    def draw_l_system(dwg, instructions, angle, length, position, current_angle):
        x, y = position
        stack = []

        for char in instructions:
            if char == "F":
                new_x = x + length * math.cos(current_angle)
                new_y = y + length * math.sin(current_angle)
                path = dwg.path(d=f"M {x},{y} A {length/2},{length/2} 0 0,1 {new_x},{new_y}", stroke='black', fill='none')
                dwg.add(path)
                x, y = new_x, new_y
            elif char == "+":
                current_angle += angle
            elif char == "-":
                current_angle -= angle
            elif char == "[":
                stack.append((x, y, current_angle))
            elif char == "]":
                x, y, current_angle = stack.pop()

    instructions = axiom
    for _ in range(iterations):
        instructions = apply_rules(instructions)

    draw_l_system(dwg, instructions, angle, length, start_position, start_angle)


def create_random_design(svg_output_dir, filename="random_design.svg"):
    width, height = PLOTTER_WIDTH, PLOTTER_HEIGHT  # Adjust width and height for margins
    dwg = svgwrite.Drawing(os.path.join(svg_output_dir, filename), size=(f'{PLOTTER_WIDTH-50}mm', f'{PLOTTER_HEIGHT-50}mm'))
    # Add multiple independent L-systems
    for _ in range(random.randint(1, 3)):  # Number of L-systems to draw
        axiom = "F-F+F+F-F+F"
        rules = {"F": "F-F-F-F+F"}  # Simplified rule for less complexity
        angle = math.pi / 2
        start_position = (random.uniform(MARGIN, width - MARGIN), random.uniform(MARGIN, height - MARGIN))
        create_l_system(dwg, axiom, rules, angle, length=random.uniform(3, 35), iterations=1, start_position=start_position, start_angle=random.uniform(0, 2 * math.pi))


    dwg.save()



def draw_random_svg():
    svg_directory = 'c:/Users/kilia/Desktop/COOP/Thesis/pre_svg'
    svg_files = [f for f in os.listdir(svg_directory) if f.endswith('.svg')]

    if not svg_files:
        print("No SVG files found in the directory.")
        return

    random_svg = random.choice(svg_files)
    svg_file_path = os.path.join(svg_directory, random_svg)
    gcode = svg_to_gcode(svg_file_path, scale=0.4, feed_rate=12000)

    # Move to random start position
    random_start_x, random_start_y = move_to_random_position()
    
    pen_down()

    # Adjust G-code to start from the random position
    adjusted_gcode = []
    for line in gcode:
        if line.startswith("G0") or line.startswith("G1"):
            parts = line.split()
            x_val = None
            y_val = None
            for part in parts:
                if part.startswith("X"):
                    x_val = float(part[1:])
                elif part.startswith("Y"):
                    y_val = float(part[1:])
            
            if x_val is not None and y_val is not None:
                new_x = x_val + random_start_x
                new_y = y_val + random_start_y
                adjusted_gcode.append(f"{parts[0]} X{new_x:.2f} Y{new_y:.2f} {' '.join(parts[3:])}")
            else:
                adjusted_gcode.append(line)
        else:
            adjusted_gcode.append(line)
    adjusted_gcode.append("M3 S50")  # Lift the pen
    adjusted_gcode.append("G0 X0 Y0")  # Move back to origin
    for line in adjusted_gcode:
        send_gcode(line.strip())
    print(f"G-code for random SVG '{random_svg}' sent to the robot at random position ({random_start_x:.2f}, {random_start_y:.2f}).")

def move_to_random_position():
    max_x = PLOTTER_WIDTH - 3 * MARGIN
    max_y = PLOTTER_HEIGHT - 3 * MARGIN
    random_start_x = random.uniform(MARGIN, max_x)
    random_start_y = random.uniform(MARGIN, max_y)
    # Move the plotter to the random starting position
    pen_up()
    send_gcode(f"G0 X{random_start_x:.2f} Y{random_start_y:.2f}")
    return random_start_x, random_start_y

def draw_two_random_sketches():
    if not categories:
        print("No categories available.")
        return

    selected_categories = random.sample(categories, 1)
    print(f"Selected categories: {selected_categories}")

    for category in selected_categories:
        draw_result(category)
        time.sleep(1)  # Add a slight delay between drawings if necessary

    
def on_press(key):
    global current_pen_position, transformation_matrix, current_frame, top_prediction_index

    try:
        key_char = key.char
    except AttributeError:
        return  # Ignore non-character keys

    print(f"Key pressed: {key_char}")  # Debugging print statement

    if key_char == 'a':
        if current_frame is not None:
            print("Frame captured. Sending to prediction model...")
            saved_corners = load_corners()
            if saved_corners:
                cropped_frame = crop_to_paper(current_frame, saved_corners)
                top_prediction_index, top_prediction_label, top_prediction_probability = predict_drawing(cropped_frame)
                print(f'Top prediction: {top_prediction_label} with probability {top_prediction_probability:.2f}')
                draw_result(top_prediction_label)
            else:
                print("No saved corners available for cropping.")
        else:
            print("No frame available to capture.")

    elif key_char == 'c':
        print("Drawing two random sketches from dataset...")
        draw_two_random_sketches()

    elif key_char == 'p':
        create_random_design(svg_output_dir)
        draw_random_svg()
        print("Random SVG drawing initiated.")
    elif key_char == '0':
        pen_up()
        send_gcode("G0 X0 Y0")  # Move to origin
        print("Moved to origin.")
        



def move_to_corner(corner_index):
    if transformation_matrix is not None:
        # Define the corners of the paper in order starting from top-right and moving clockwise
        plotter_corners = np.array([
            [0, 0],  # top-right
            [0, PLOTTER_HEIGHT],  # bottom-right
            [PLOTTER_WIDTH, PLOTTER_HEIGHT],  # bottom-left
            [PLOTTER_WIDTH, 0]  # top-left
        ])

        # Use the predefined plotter corners directly
        plotter_corner = plotter_corners[corner_index]

        # Debugging: Print the plotter coordinates to verify correctness
        print(f"Moving to Corner {corner_index + 1}: {plotter_corner}")

        # Move the plotter to the specified corner
        pen_up()
        send_gcode(generate_gcode(plotter_corner[0], plotter_corner[1]))
        pen_down()

        
def camera_to_plotter(point, transformation_matrix):
    point_homogeneous = np.array([point[0], point[1], 1]).reshape((3, 1))
    transformed_point_homogeneous = np.dot(transformation_matrix, point_homogeneous).flatten()
    transformed_x, transformed_y = transformed_point_homogeneous[:2]

    # Ensure the coordinates are within the plotter's bounds
    adjusted_x = max(0, min(transformed_x, PLOTTER_WIDTH))
    adjusted_y = max(0, min(transformed_y, PLOTTER_HEIGHT))

    return int(adjusted_x), int(adjusted_y)




def click_event(event, x, y, flags, param):
    global transformation_matrix
    if event == cv2.EVENT_LBUTTONDOWN:
        if transformation_matrix is not None:
            plotter_coords = camera_to_plotter((x, y), transformation_matrix)
            print(f"Clicked at: ({x}, {y}), move plotter to: {plotter_coords}")
            send_gcode(generate_gcode(plotter_coords[0], plotter_coords[1]))
            print(f"Sent G-code to move plotter to: {plotter_coords}")




def detect_shapes_on_paper(frame, corners, margin=MARGIN, threshold_val=150):
    global detected_shapes
    detected_shapes.clear()
    
    # Create a mask to focus on the area within the corners with a margin
    mask = np.zeros_like(frame, dtype=np.uint8)
    paper_contour = np.array(corners)
    paper_contour_with_margin = np.array([
        [paper_contour[0][0] + margin, paper_contour[0][1] + margin],
        [paper_contour[1][0] + margin, paper_contour[1][1] - margin],
        [paper_contour[2][0] - margin, paper_contour[2][1] - margin],
        [paper_contour[3][0] - margin, paper_contour[3][1] + margin]
    ])
    cv2.drawContours(mask, [paper_contour_with_margin], -1, (255, 255, 255), -1)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to get the black lines on white paper
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply the mask to the thresholded image
    thresh = cv2.bitwise_and(thresh, mask[:, :, 0])
    
    # Apply morphological operations to enhance the pen lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area to remove large white areas
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 10000:  # Minimum and maximum area thresholds
            smoothed_hull = smooth_contour(contour, smoothing_factor=0.002)
            detected_shapes.append(smoothed_hull)

    # Display the thresholded frame for debugging
    cv2.imshow('Thresholded Frame', thresh)

    return thresh







def smooth_contour(contour, smoothing_factor=0.002):
    epsilon = smoothing_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def prepare_gcode_sequence(detected_shapes, transformation_matrix):
    gcode_sequence = ["M3 S50\n"]
    for shape in detected_shapes:
        for point in shape[1:]:
            transformed_point = camera_to_plotter((point[0][0], point[0][1]), transformation_matrix)
            gcode_sequence.append(generate_gcode(transformed_point[0], transformed_point[1]))
    gcode_sequence.append("M3 S0\n")
    gcode_sequence.append("G0 X0 Y0\n")
    return gcode_sequence

def main():
    global paper_contour, transformation_matrix, corner_points, current_pen_position, current_frame, saved_corners

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use index 0 since it works
    if not cap.isOpened():
        print("Failed to open camera.")
        return
    else:
        print("Camera opened successfully.")

    # Load saved corners
    saved_corners = load_corners()

    # Capture frame for calibration
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        cap.release()
        return

    if saved_corners:
        print("Using saved corners:", saved_corners)
        transformation_matrix = apply_transformation_matrix(frame, saved_corners)
        if transformation_matrix is not None:
            print("Calibration Successful with saved corners. Transformation Matrix:", transformation_matrix)
        else:
            print("Failed to apply transformation matrix with saved corners.")
            cap.release()
            return

    visualize_calibration(frame, saved_corners if saved_corners else corner_points, transformation_matrix)

    # Set camera resolution
    desired_width = 1280
    desired_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("Camera Online")

    create_trackbar()  # Initialize the trackbar

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame, retrying...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("Failed to open camera on retry.")
                return
            continue
        current_frame = frame

        # Get the current threshold value from the trackbar
        threshold_val = cv2.getTrackbarPos('Threshold', 'Thresholded Frame')

        # Detect shapes with the current threshold value
        thresh = detect_shapes_on_paper(frame, saved_corners, threshold_val=threshold_val)
        detect_lines(current_frame, threshold_val=threshold_val)
        cv2.imshow('Master Thesis Vessi', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Master Thesis Vessi', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    closeSystem()
    listener.stop()
    cv2.destroyAllWindows()

# def main():
#     global paper_contour, transformation_matrix, corner_points, current_pen_position, current_frame

#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use index 0 since it works
#     if cap.isOpened():
#         print("Camera opened successfully.")
#     else:
#         print("Failed to open camera.")
#         return


#     # Load saved corners
#     saved_corners = load_corners()

#     # Capture frame for calibration
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         cap.release()
#         return

#     if saved_corners:
#         print("Using saved corners:", saved_corners)
#         transformation_matrix = apply_transformation_matrix(frame, saved_corners)
#         if transformation_matrix is not None:
#             print("Calibration Successful with saved corners. Transformation Matrix:", transformation_matrix)
#         else:
#             print("Failed to apply transformation matrix with saved corners.")
#             cap.release()
#             return

#     visualize_calibration(frame, saved_corners if saved_corners else corner_points, transformation_matrix)

#     # Set camera resolution
#     desired_width = 1280
#     desired_height = 720
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     print("Camera Online")
    
#     create_trackbar()  # Initialize the trackbar


#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("Failed to grab frame, retrying...")
#             cap.release()
#             time.sleep(1)
#             cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#             if not cap.isOpened():
#                 print("Failed to open camera on retry.")
#                 return
#             continue
#         current_frame = frame
#         threshold_val = cv2.getTrackbarPos('Threshold', 'Thresholded Frame')

#         detect_shapes_on_paper(frame, saved_corners, threshold_val=threshold_val)

        
#         if saved_corners is not None:
#             cv2.drawContours(frame, [np.array(saved_corners)], -1, (0, 255, 0), 3)
#             for shape in detected_shapes:
#                 cv2.polylines(frame, [shape], isClosed=True, color=(255, 0, 0), thickness=2)

    


#         cv2.imshow('Master Thesis Vessi', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Master Thesis Vessi', cv2.WND_PROP_VISIBLE) < 1:
#             break

#     cap.release()
#     closeSystem()
#     listener.stop()
#     cv2.destroyAllWindows()



# Function to clean up system and release resources
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
    ser.close()
    sender_thread.join()
    print("Camera released and windows closed and Plotter at Origin. Cleanup complete.")

# Start the listener for keyboard events
listener = keyboard.Listener(on_press=on_press)
listener.start()

if __name__ == "__main__":
    main()
