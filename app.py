from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('hand_gesture_model.keras')
class_labels = ['play_pause','volume_up', 'volume_down', 'forward', 'backward']

def preprocess_image(image_bytes):
    # Decode image bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess like your original script
    imgsize = 300
    imgwhite = np.ones((imgsize,imgsize,3),np.uint8)*255
    
    h, w, _ = img.shape
    aspectratio = h / w
    
    if aspectratio > 1:
        k = imgsize / h
        wcal = int(k * w)
        imgresize = cv2.resize(img, (wcal, imgsize))
        wgap = (imgsize - wcal) // 2
        imgwhite[:, wgap:wcal+wgap] = imgresize
    else:
        k = imgsize / w
        hcal = int(k * h)
        imgresize = cv2.resize(img, (imgsize, hcal))
        hgap = (imgsize - hcal) // 2
        imgwhite[hgap:hcal+hgap, :] = imgresize
    
    imgwhite = imgwhite / 255.0
    imgwhite = np.expand_dims(imgwhite, axis=0)
    return imgwhite

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    processed_img = preprocess_image(img_bytes)
    predictions = model.predict(processed_img)
    predicted_class = class_labels[np.argmax(predictions)]
    
    return jsonify({'gesture': predicted_class})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
