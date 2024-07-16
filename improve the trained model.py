import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Input
import random

# Paths
data_dir = r'C:\Users\user\AppData\Local\Google\Cloud SDK\quickdraw_dataset\quickdraw_dataset'
model_dir = os.path.join(data_dir, 'model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'quickdraw_model.keras')
categories_file = r'C:\Users\user\AppData\Local\Google\Cloud SDK\quickdraw_dataset\categories.txt'

# Load the categories
def load_categories(file_path):
    print(f"Loading categories from {file_path}")
    with open(file_path, 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(categories)} categories.")
    return categories

categories = load_categories(categories_file)
num_classes = len(categories)

# Function to parse NDJSON files with random sampling
def parse_ndjson(file_path, max_samples):
    print(f"Parsing NDJSON file {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Shuffle the lines for random sampling
    random.shuffle(lines)
    
    drawings = []
    for i, line in enumerate(lines):
        if i >= max_samples:
            break
        try:
            data = json.loads(line.strip())
            drawing = data['drawing']
            simplified_drawing = []
            for stroke in drawing:
                simplified_drawing.extend(zip(stroke[0], stroke[1]))
            drawings.append(simplified_drawing)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error decoding JSON on line: {line.strip()}")
            print(f"Error message: {e}")
    print(f"Parsed {len(drawings)} drawings from {file_path}")
    return drawings

# Load dataset from NDJSON files
def load_data_from_ndjson(data_dir, categories, max_samples_per_category):
    X, y = [], []
    max_length = 0

    for label, category in enumerate(categories):
        print(f"Processing category '{category}' with label {label}")
        category_file = os.path.join(data_dir, category + '.ndjson')
        if not os.path.exists(category_file):
            print(f"File {category_file} does not exist. Skipping category '{category}'.")
            continue
        drawings = parse_ndjson(category_file, max_samples_per_category)
        for drawing in drawings:
            x_points = [point[0] for point in drawing]
            y_points = [point[1] for point in drawing]
            lengths = list(range(len(drawing)))

            length = len(drawing)
            max_length = max(max_length, length)

            x_points = np.array(x_points, dtype=np.float32)
            y_points = np.array(y_points, dtype=np.float32)
            lengths = np.array(lengths, dtype=np.float32)

            features = np.stack([x_points, y_points, lengths], axis=1)
            X.append(features)
            y.append(label)

    # Pad sequences to the same length
    print(f"Padding sequences to the same length: {max_length}")
    X_padded = []
    for sequence in X:
        padded_sequence = np.pad(sequence, ((0, max_length - sequence.shape[0]), (0, 0)), 'constant')
        X_padded.append(padded_sequence)

    print(f"Loaded data for {len(X_padded)} samples.")
    return np.array(X_padded), np.array(y, dtype=np.int64), max_length

print("Loading data from NDJSON files...")

max_samples_per_category = 1500
X, y, max_length = load_data_from_ndjson(data_dir, categories, max_samples_per_category)

# Split data into training and validation sets for each category
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = [], [], [], []
split_ratio = 0.8

for i in range(num_classes):
    indices = np.where(y == i)[0]
    np.random.shuffle(indices)
    split_index = int(len(indices) * split_ratio)
    X_train.extend(X[indices[:split_index]])
    y_train.extend(y[indices[:split_index]])
    X_val.extend(X[indices[split_index:]])
    y_val.extend(y[indices[split_index:]])

X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Load existing model
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    model = load_model(model_path)
else:
    # Model architecture
    def build_crnn_model(input_shape, num_classes):
        print("Building the CRNN model...")
        model = Sequential([
            Input(shape=input_shape),
            Masking(mask_value=0.),
            LSTM(128, return_sequences=True),
            LSTM(128),
            Dense(256, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        return model

    # Build and compile the model with a flexible input shape
    input_shape = (max_length, 3)  # Variable length sequence with 3 features
    model = build_crnn_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

# Continue training the model
print("Continuing the training of the model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
print(f"Saving the model to {model_path}...")
model.save(model_path)
print("Model saved successfully.")
