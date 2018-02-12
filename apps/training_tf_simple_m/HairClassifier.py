import cv2
import numpy as np
import robolib.modelmanager.downloader as downloader

MODEL_FILE = 'FrontalFace.xml'
downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, MODEL_FILE, True)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, (40, 40), (100, 100), True)
    if len(faces) > 0:
        x, y, w, h = faces[np.argmax(levelWeights)]
        face = img[y:y+h+1, x:x+w+1]
        resImg = cv2.resize(face, dst=None, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('img', resImg)
        cv2.imshow('face', face)
        print(resImg)
