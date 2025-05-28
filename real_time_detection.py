import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from tensorflow.keras.models import load_model
import pyautogui

import tensorflow as tf
print(tf.__version__)


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset =20
imgsize =300
counter =0

# VLC media player control functions
# VLC media player control functions
def play_pause():
    pyautogui.press('space')
    time.sleep(1)  # Introduce a 1-second delay

def volume_up():
    pyautogui.hotkey('ctrl', 'up')


def volume_down():
    pyautogui.hotkey('ctrl', 'down')


def forward():
    pyautogui.press('right')


def backward():
    pyautogui.press('left')
    time.sleep(1)  # Introduce a 1-second delay

# Load the trained model
model = load_model('hand_gesture_model1.h5')

# Real-time detection using webcam
cap = cv2.VideoCapture(0)

# Define classes
class_labels = ['play_pause','volume_up', 'volume_down', 'forward', 'backward']



while True:
    sucess,img = cap.read()
    imgoutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgwhite = np.ones((imgsize,imgsize,3),np.uint8)*255
        imgCrop = img[y-offset:y+offset+h,x-offset:x+offset+w]

        imgcropshape = imgCrop.shape



        aspectratio = h/w

        if aspectratio>1:
            k= imgsize / h
            wcal = math.ceil(k*w)
            imgresize = cv2.resize(imgCrop,(wcal,imgsize))
            imgresizeshape = imgresize.shape
            wgap = math.ceil((imgsize-wcal)/2)
            imgwhite[:, wgap:wcal+wgap] = imgresize

        else:
            k= imgsize / w
            hcal = math.ceil(k*h)
            imgresize = cv2.resize(imgCrop,(imgsize,hcal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize-hcal)/2)
            imgwhite[hgap:hcal+hgap, :] = imgresize

        # Make predictions using the trained model
        predictions = model.predict(np.expand_dims(imgwhite, axis=0))

        # Print the predicted matrix
        print("Predicted Matrix:")
        print(np.array2string(predictions, precision=2, separator=', ', suppress_small=True))

        # Get the predicted class
        predicted_class = class_labels[np.argmax(predictions)]

        # Display the result on the frame
        cv2.putText(imgoutput, f'Prediction: {predicted_class}', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(predicted_class)

        # Perform actions based on predictions
        if predicted_class == 'play_pause':
            play_pause()
        elif predicted_class == 'volume_up':
            volume_up()
        elif predicted_class == 'volume_down':
            volume_down()
        elif predicted_class == 'forward':
            forward()
        elif predicted_class == 'backward':
            backward()

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imagewhite",imgwhite)
    cv2.imshow("Image", imgoutput)
    cv2.waitKey(1)
