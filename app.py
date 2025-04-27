from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Initialize app
app = Flask(__name__)

# Load model
model = load_model("Skin_disease_Predictor_model.keras")
IMG_SIZE = (128, 128)
categories = ['Vitiligo', 'Rashes', 'Lupus', 'Acne']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded file
        file = request.files['image']
        if file:
            filepath = os.path.join('static', file.filename)
            file.save(filepath)

            # Preprocess image
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img_input = np.expand_dims(img / 255.0, axis=0)

            # Predict
            prediction = model.predict(img_input)
            predicted_class_idx = np.argmax(prediction)
            predicted_class_label = categories[predicted_class_idx]

            return render_template('index.html', prediction=predicted_class_label, image_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
