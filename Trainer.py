#author Kilian Vesshoff< 22-03-2024, Creating a Dataset for YOLO model training
import cv2
import numpy as np
import os
from pathlib import Path
import yaml  # Ensure PyYAML is installed (`pip install PyYAML`)

def convert_bbox_to_yolo_format(bbox, image_shape):
    dw = 1. / image_shape[1]
    dh = 1. / image_shape[0]
    x_center = (bbox[0] + bbox[2] / 2.0) * dw
    y_center = (bbox[1] + bbox[3] / 2.0) * dh
    w = bbox[2] * dw
    h = bbox[3] * dh
    return (x_center, y_center, w, h)

def save_yolo_annotation(filename, bbox, image_shape):
    bbox_yolo = convert_bbox_to_yolo_format(bbox, image_shape)
    with open(filename, 'w') as file:
        file.write(f"0 {bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]}\n")

def setup_dataset_structure(base_path):
    paths = {
        "images_train": os.path.join(base_path, "images", "train"),
        "images_val": os.path.join(base_path, "images", "val"),
        "labels_train": os.path.join(base_path, "labels", "train"),
        "labels_val": os.path.join(base_path, "labels", "val"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def create_yaml_file(base_path, num_classes=1, class_names=["pen"]):
    data = {
        "train": "images/train",
        "val": "images/val",
        "nc": num_classes,
        "names": class_names
    }
    with open(os.path.join(base_path, "coco8.yaml"), 'w') as yamlfile:
        yaml.dump(data, yamlfile, default_flow_style=False)

# Main script starts here
dataset_base_path = str(Path.home() / "Desktop/pen_datasetBIG")
paths = setup_dataset_structure(dataset_base_path)  # Define paths for dataset structure

# Initialize webcam
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow('Webcam', frame)
    key = cv2.waitKey(1)

    # Press 's' for training data, 'v' for validation data
    if key in [ord('s'), ord('v')]:
        is_train = key == ord('s')
        screenshot = frame.copy()
        bbox = cv2.selectROI('Webcam', screenshot, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow('Webcam')

        image_id = len(os.listdir(paths["images_train" if is_train else "images_val"])) + 1
        image_name = f"pen_{image_id}.jpg"
        label_name = f"pen_{image_id}.txt"

        image_path = os.path.join(paths["images_train" if is_train else "images_val"], image_name)
        label_path = os.path.join(paths["labels_train" if is_train else "labels_val"], label_name)

        cv2.imwrite(image_path, screenshot)
        save_yolo_annotation(label_path, bbox, frame.shape)

        print("Image and annotation saved successfully!")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# After collecting data, create the YAML file
create_yaml_file(dataset_base_path)  # This will create "coco8.yaml" with the dataset configuration
