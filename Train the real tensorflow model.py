import os
import numpy as np
import tensorflow as tf
import ndjson
import cv2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Directory containing .ndjson files
data_dir = r'C:\Users\user\AppData\Local\Google\Cloud SDK\quickdraw_dataset\quickdraw_dataset'
# Directory to save the trained model
model_dir = os.path.join(data_dir, 'model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'quickdraw_model.keras')

# Path to the categories file
categories_file = r'C:\Users\user\AppData\Local\Google\Cloud SDK\quickdraw_dataset\categories.txt'

# Function to load categories from file
def load_categories(file_path):
    with open(file_path, 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    return {category: idx for idx, category in enumerate(categories)}

label_dict = load_categories(categories_file)

# Function to parse drawing and create an image
def parse_drawing(drawing):
    image = np.zeros((28, 28), dtype=np.uint8)
    for stroke in drawing:
        for i in range(len(stroke[0]) - 1):
            x1, y1 = stroke[0][i], stroke[1][i]
            x2, y2 = stroke[0][i + 1], stroke[1][i + 1]
            image = cv2.line(image, (x1, y1), (x2, y2), 255, 1)
    return image

# Generator function to yield data in batches
def data_generator(data_dir, label_dict, batch_size=32, samples_per_category=50, shuffle=True):
    files = [f for f in os.listdir(data_dir) if f.endswith('.ndjson')]
    if shuffle:
        np.random.shuffle(files)
    
    while True:
        X, y = [], []
        for file in files:
            category = file.split('.')[0]
            if category in label_dict:
                with open(os.path.join(data_dir, file)) as f:
                    data = ndjson.load(f)
                    indices = np.arange(len(data))
                    if shuffle:
                        np.random.shuffle(indices)
                    for i in indices[:samples_per_category]:
                        drawing = data[i]
                        image = parse_drawing(drawing['drawing'])
                        X.append(image)
                        y.append(label_dict[category])
                        if len(X) == batch_size:
                            yield np.array(X).reshape(-1, 28, 28, 1).astype(np.float32) / 255.0, np.array(y)
                            X, y = [], []

        if X and y:
            yield np.array(X).reshape(-1, 28, 28, 1).astype(np.float32) / 255.0, np.array(y)

# Custom callback to log end of epoch
class EpochEndLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"End of epoch {epoch + 1}, logs: {logs}")

# Create the model
def create_model(num_classes):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Load data
logging.info("Loading data...")
batch_size = 32
samples_per_category = 10  # Number of samples per category to load for training per epoch
val_samples_per_category = 5  # Number of samples per category to load for validation per epoch

# Create datasets
train_gen = data_generator(data_dir, label_dict, batch_size=batch_size, samples_per_category=samples_per_category, shuffle=True)
val_gen = data_generator(data_dir, label_dict, batch_size=batch_size, samples_per_category=val_samples_per_category, shuffle=False)

# Count steps for training and validation
train_steps = (samples_per_category * len(label_dict)) // batch_size
val_steps = (val_samples_per_category * len(label_dict)) // batch_size

# Debugging: Print steps per epoch and validation steps
logging.info(f"Training steps per epoch: {train_steps}, Validation steps: {val_steps}")

# Create and compile the model
num_classes = len(label_dict)
model = create_model(num_classes)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)
model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
epoch_end_logger = EpochEndLogger()

# Train the model
logging.info("Starting training...")
try:
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=5,  # Reduce the number of epochs for quicker testing
        steps_per_epoch=train_steps, 
        validation_steps=val_steps, 
        callbacks=[early_stopping, model_checkpoint, epoch_end_logger]
    )
    logging.info("Training completed successfully.")
except Exception as e:
    logging.error(f"An error occurred during training: {e}")

# Save the final model if not already saved
if not os.path.exists(model_path):
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
else:
    logging.info("Model checkpoint already saved the best model.")
