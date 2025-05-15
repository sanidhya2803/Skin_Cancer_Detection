from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import keras

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/'  # Folder to store uploaded images

# Load the model
model = tf.keras.models.load_model('melanoma_classification_model.h5')

def predict_skin_cancer(image_path):
    # Load the image and preprocess it for prediction
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict the class probabilities (output from the model)
    prediction = model.predict(img_array)
    
    # For binary classification (Benign/Malignant)
    if prediction.shape[-1] == 1:  # Binary classification output
        predicted_class = 'Malignant' if prediction[0] > 0.5 else 'Benign'
        confidence = prediction[0][0] * 100  # For binary, prediction is a single value (0 or 1)
    else:
        # For multi-class classification (use softmax output)
        class_names = ['Benign', 'Malignant']  # Update with your actual class names
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100  # Get the highest probability
    
    return predicted_class, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict class and confidence
        prediction, confidence = predict_skin_cancer(filepath)
        
        # Pass the image URL and prediction results to the template
        image_url = filename
        return render_template('result.html', prediction=prediction, confidence=confidence, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
