import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_datasets as tfds
import ndjson
import cv2

# Directory to save the trained model
model_dir = r'C:\Users\user\AppData\Local\Google\Cloud SDK\quickdraw_dataset'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'quickdraw_model.keras')

# Path to the categories file
categories_file = os.path.join(model_dir, 'categories.txt')

# Path to the dataset files
dataset_dir = os.path.join(model_dir, 'quickdraw_dataset')

# Function to load categories from file
def load_categories(file_path):
    with open(file_path, 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    return categories

categories = load_categories(categories_file)

# Function to convert strokes to a 28x28 image
def strokes_to_image(strokes, size=28):
    image = np.zeros((256, 256), np.uint8)
    for stroke in strokes:
        for i in range(len(stroke[0]) - 1):
            x1, y1 = stroke[0][i], stroke[1][i]
            x2, y2 = stroke[0][i+1], stroke[1][i+1]
            cv2.line(image, (x1, y1), (x2, y2), 255, 2)
    image = cv2.resize(image, (size, size))
    return image

# Function to load Quick, Draw! dataset from .ndjson files
def load_quickdraw_data(dataset_dir, categories):
    data = []
    labels = []
    for idx, category in enumerate(categories):
        file_path = os.path.join(dataset_dir, f'{category}.ndjson')
        with open(file_path, 'r') as f:
            drawings = ndjson.load(f)
            for drawing in drawings:
                strokes = drawing['drawing']
                image = strokes_to_image(strokes)
                data.append(image)
                labels.append(idx)
    return np.array(data), np.array(labels)

# Load the dataset
data, labels = load_quickdraw_data(dataset_dir, categories)
data = data.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0

# Split the dataset into training and validation sets
split_index = int(0.9 * len(data))
x_train, x_val = data[:split_index], data[split_index:]
y_train, y_val = labels[:split_index], labels[split_index:]

# Function to evaluate the model
def evaluate_model(model, ds_val):
    _, accuracy = model.evaluate(ds_val)
    print(f'Validation accuracy: {accuracy * 100:.2f}%')
    return accuracy

# Function to train the model until it reaches 90% accuracy
def train_until_90_percent(model, ds_train, ds_val):
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

    while True:
        print("Starting training epoch...")
        model.fit(ds_train, validation_data=ds_val, epochs=1, callbacks=[early_stopping, model_checkpoint])
        accuracy = evaluate_model(model, ds_val)
        if accuracy >= 0.90:
            print("Achieved 90% accuracy. Stopping training.")
            break
    model.load_weights(model_path)  # Load the best model
    print(f"Model saved to {model_path}")

# Data augmentation using TensorFlow functions
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

# Create TensorFlow datasets
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

# Apply data augmentation
ds_train = ds_train.map(augment).batch(100).prefetch(tf.data.experimental.AUTOTUNE)
ds_val = ds_val.batch(100).prefetch(tf.data.experimental.AUTOTUNE)

# Check if the model file exists
if os.path.exists(model_path):
    print("Loading existing model...")
    model = tf.keras.models.load_model(model_path)
else:
    print("Model not found. Starting training from scratch...")
    num_classes = len(categories)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Added dropout for regularization
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model until it reaches 90% accuracy
train_until_90_percent(model, ds_train, ds_val)
