import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import pandas as pd

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset =20
imgsize =300
counter =0
folder = r'C:\Users\PARAM M. SURELIYA\PycharmProjects\sign language\Data\backward'
while True:
    sucess,img = cap.read()
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

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imagewhite",imgwhite)
    cv2.imshow("Image", img)
    key= cv2.waitKey(1)
    if key ==ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(counter)