from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('quickdraw_model.h5')

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    expanded = np.expand_dims(normalized, axis=-1)
    expanded = np.expand_dims(expanded, axis=0)
    return expanded

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_base64 = data.get('image')
    
    if img_base64 is None:
        return jsonify({"error": "No image provided"}), 400
    
    # Decode the base64 image
    img_bytes = base64.b64decode(img_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(img)
    
    # Predict using the model
    predictions = model.predict(preprocessed_image)
    top_prediction = np.argmax(predictions[0])
    
    return jsonify({"prediction": int(top_prediction)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
