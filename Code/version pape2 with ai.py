import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from pynput import keyboard
import serial
from serial import Serial
import time
import threading
import queue
from queue import Queue
from shapely.geometry import LineString, Polygon, MultiLineString
import random
import shapely

# Load calibration data
with np.load('c:/Users/kilia/Desktop/COOP/Thesis/calibration_data.npz') as X:
    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

ser = serial.Serial('COM4', 115200, timeout=1)
transformation_matrix = None

# Check if the model file exists
model_path = 'quickdraw_model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
else:
    # Load the Quick, Draw! dataset
    ds_train = tfds.load('quickdraw_bitmap', split='train', as_supervised=True)
    ds_train = ds_train.take(80000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    # Check the number of classes
    num_classes = tfds.builder('quickdraw_bitmap').info.features['label'].num_classes
    
    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Adjust output layer to match num_classes
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(ds_train, epochs=10)
    
    # Save the model
    model.save(model_path)

# Function to preprocess the image and predict using the model
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    expanded = np.expand_dims(normalized, axis=-1)
    expanded = np.expand_dims(expanded, axis=0)
    return expanded

def predict_drawing(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    top_prediction = np.argmax(predictions[0])
    return top_prediction

# Store calibration points
# Global variables for calibration
gcode_queue = queue.Queue()  # for G-code send
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

# in mm
PLOTTER_WIDTH = 210  # A4 paper width in mm
PLOTTER_HEIGHT = 297  # A4 paper height in mm

def detect_paper(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find the largest contour, which we will assume to be the paper
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) == 4:
        return approx
    else:
        return None

def detect_markers_and_calibrate(frame):
    paper_contour = detect_paper(frame)
    if paper_contour is None:
        print("Error: Paper not detected.")
        return None, paper_contour
    
    # Sort the points to align with the plotter's coordinate system
    paper_contour = sorted(paper_contour.reshape(4, 2), key=lambda x: (x[1], x[0]))  # Sort by y first, then by x
    top_left, top_right, bottom_left, bottom_right = paper_contour
    
    # Define camera points and corresponding plotter points
    camera_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    plotter_points = np.array([
        [PLOTTER_WIDTH, 0],  # top-right in plotter space
        [0, 0],  # top-left in plotter space
        [0, PLOTTER_HEIGHT],  # bottom-left in plotter space
        [PLOTTER_WIDTH, PLOTTER_HEIGHT]  # bottom-right in plotter space
    ], dtype="float32")

    # Calculate the transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(camera_points, plotter_points)
    return transformation_matrix, paper_contour

def visualize_calibration(frame, paper_contour, transformation_matrix):
    # Visualize paper contour
    if paper_contour is not None:
        cv2.polylines(frame, [paper_contour], isClosed=True, color=(0, 255, 0), thickness=2)
    
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

def calibrate_plotter_camera_system(cap):
    # Capture frame for calibration
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame for calibration")
        return None    
    frame_undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
    transformation_matrix, paper_contour = detect_markers_and_calibrate(frame_undistorted)
    
    if transformation_matrix is None:
        print("Calibration Failed.")
        return None
    
    visualize_calibration(frame_undistorted, paper_contour, transformation_matrix)
    return transformation_matrix

# G-Code Functions
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
            pass

def send_gcode(command):
    gcode_queue.put(command + "\n")

def pen_up():
    send_gcode("M3 S50")  # Pen up command

def pen_down():
    send_gcode("M3 S0")  # Pen down command for drawing

# Start the G-Code sender thread
sender_thread = threading.Thread(target=gcode_sender, daemon=True)
sender_thread.start()   

def generate_gcode(x_plot, y_plot, feed_rate=10000):
    move_command = f"G01 X{x_plot:.2f} Y{y_plot:.2f} F{feed_rate}"
    return f"{move_command}"

def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)  # Use the appropriate camera index (0 for default)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):  # Press space to capture the frame
            captured_image = frame
            break
    cap.release()
    cv2.destroyAllWindows()
    return captured_image

def on_press(key):
    if key == keyboard.Key.space:
        captured_image = capture_image_from_webcam()
        top_prediction = predict_drawing(captured_image)
        print(f'Top prediction: {top_prediction}')

def camera_to_plotter(point, transformation_matrix):
    global scaling_factor
    point_homogenous = np.array([point[0], point[1], 1]).reshape((3, 1))
    transformed_point_homogenous = np.dot(transformation_matrix, point_homogenous).flatten()
    transformed_x, transformed_y = transformed_point_homogenous[:2]
    adjusted_x = transformed_x * scaling_factor
    adjusted_y = transformed_y * scaling_factor
    print(f"Original: {point}, Transformed: ({adjusted_x}, {adjusted_y})")
    x_plot_adjusted = max(0, min(adjusted_x, drawable_area_width))
    y_plot_adjusted = max(0, min(adjusted_y, drawable_area_height))
    return int(x_plot_adjusted), int(y_plot_adjusted)

def clip_line_to_contour(start_point, end_point, contour_polygon):
    line = LineString([start_point, end_point])
    clipped_line = line.intersection(contour_polygon)
    if not clipped_line.is_empty:
        if isinstance(clipped_line, LineString):
            return [np.array(clipped_line.coords)]
        elif isinstance(clipped_line, MultiLineString):
            return [np.array(line.coords) for line in clipped_line.geoms]
    return None

def draw_line(start_pt, end_pt):
    send_gcode(generate_gcode(start_pt[0], start_pt[1]))
    pen_down()
    send_gcode(generate_gcode(end_pt[0], end_pt[1]))
    pen_up()

def detect_shapes_on_paper(frame, paper_contour):
    global detected_shapes
    detected_shapes.clear()
    if paper_contour is not None:
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.drawContours(mask, [paper_contour], -1, (255, 255, 255), -1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, mask[:, :, 0])
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            hull = cv2.convexHull(contour)
            smoothed_hull = smooth_contour(hull, smoothing_factor=0.002)
            if cv2.contourArea(smoothed_hull) > 100:
                detected_shapes.append(smoothed_hull)

def smooth_contour(contour, smoothing_factor=0.002):
    arc_length = cv2.arcLength(contour, True)
    epsilon = smoothing_factor * arc_length
    if arc_length == 0:
        return contour
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

def on_press(key):
    global detected_shapes, transformation_matrix, paper_contour
    if key == keyboard.Key.space:
        if transformation_matrix is not None and paper_contour is not None:
            random.shuffle(detected_shapes)
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

listener = keyboard.Listener(on_press=on_press)
listener.start()
paper_contour = None

def collect_gcode_commands(vor, paper_contour):
    gcode_commands = []
    gcode_commands.append("M3 S50")
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
                        gcode_commands.append("M3 S0")
                        gcode_commands.append(generate_gcode(start_plotter[0], start_plotter[1]))
                        gcode_commands.append(generate_gcode(end_plotter[0], end_plotter[1]))
                        gcode_commands.append("M3 S50")
    gcode_commands.append("G0 X0 Y0")
    return gcode_commands

def main():
    global paper_contour
    global transformation_matrix
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        cap.release()
        return
    else:
        transformation_matrix, paper_contour = detect_markers_and_calibrate(frame)
        if transformation_matrix is not None:
            print("Calibration Successful. Transformation Matrix:", transformation_matrix)
        else:
            print("Calibration Failed.")
            cap.release()
            cv2.destroyAllWindows()
            closeSystem()
            listener.stop()
            return
    cap.release()
    
    visualize_calibration(frame, paper_contour, transformation_matrix)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Camera Online")
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        paper_contour = detect_paper(frame)
        
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
    pen_down()
    time.sleep(0.1)
    ser.close()
    sender_thread.join()
    print("Camera released and windows closed and Plotter at Origin. Cleanup complete.")

if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    main()
