from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import os

app = Flask(__name__)

MODEL_PATH = os.path.join('model', 'car_motorcycle_classifier.h5')
model = load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return 'Motorcycle' if prediction[0][0] < 0.5 else 'Car'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kiểm tra file upload
        if 'file' not in request.files:
            return 'No file uploaded', 400
        file = request.files['file']

        if file.filename == '':
            return 'No file selected', 400
        
        file_path = os.path.join('static', file.filename)
        file.save(file_path)


        result = predict_image(file_path)
        return render_template('index.html', result=result, file_path=file_path)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)