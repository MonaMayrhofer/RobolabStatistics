import cv2
import numpy as np
import robolib.modelmanager.downloader as downloader

MODEL_FILE = 'FrontalFace.xml'
downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, MODEL_FILE, False)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)
cap = cv2.VideoCapture(0)
cv2.namedWindow('img')

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, (40, 40), (100, 100), True)
    for i in range(len(faces)):
        x, y, w, h = faces[i]
        face = img[y-10:y+h+11, x-10:x+w+11]
        if i == 0:
            resImg = cv2.resize(face, dst=None, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        else:
            resImg = cv2.hconcat([cv2.resize(face, dst=None, dsize=(128, 128), interpolation=cv2.INTER_LINEAR), resImg])
    if len(faces) > 0:
        cv2.imshow('img', resImg)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
