
import cv2
from ultralytics import YOLO
# Load a pretrained YOLOv8n model
model = YOLO(r'C:\Users\kilia\Desktop\COOP\model3.pt')

# Read an image using OpenCV
source = cv2.imread(r'C:\Users\kilia\Desktop\pen_dataset\images\val\pen_2.jpg')

# Run inference on the source
results = model(source)