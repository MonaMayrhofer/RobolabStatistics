import cv2
import numpy as np
import os.path
from urllib.request import urlretrieve
from threading import Thread
import random

def download_xml(url, filename):
    if os.path.isfile(filename):
        print("{} already present, to re-download it remove the file.".format(filename))
    else:
        print("{} not found, downloading it from: {} ".format(filename, url))
        urlretrieve(url, filename)


FRONTALFACE_URL = \
    'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
FRONTALFACE_FILENAME = 'FrontalFace.xml'
EYE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml'

download_xml(FRONTALFACE_URL, FRONTALFACE_FILENAME)


face_cascades = cv2.CascadeClassifier(FRONTALFACE_FILENAME)

cap = cv2.VideoCapture(0)

divSpeed = 5
frameCount = 0

minSize = (50, 50)
maxSize = (150, 150)

ret, img = cap.read()
paused = True
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, minSize, maxSize, True)


def updateCamera():
    global img
    global faces
    global paused
    global minSize
    global maxSize
    timeout = 10
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        oldFaces = faces
        faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, minSize, maxSize, True)
        if len(faces) != 2:
            if timeout > 0:
                timeout -= 1
                faces = oldFaces
            else:
                paused = True
        else:
            if timeout < 10:
                timeout += 1
            paused = False


t = Thread(target=updateCamera)
t.start()

directX = random.uniform(-0., 0.9)
direction = (directX, (1-directX) ** 0.5)
speed = 3
ballPos = (img.shape[1]/2, img.shape[0]/2)

while True:
    if not paused:
        x1, y1, w1, h1 = faces[0]
        x2, y2, w2, h2 = faces[1]
        if x1 > x2:
            x1, y1, w1, h1 = faces[1]
            x2, y2, w2, h2 = faces[0]
        z1 = x1 + w1
        t1 = y1 + h1
        z2 = x2 + w2
        t2 = y2 + h2
        if (ballPos[0] - 20 < z1 and ballPos[1] < t1 and ballPos[1] > y1 and direction[0] < 0 and ballPos[0] + 40 > z1)\
                or (ballPos[0] + 20 > x2 and ballPos[1] < t2 and ballPos[1] > y2 and direction[0] > 0\
                    and ballPos[0] - 40 < z2):
            direction = (-direction[0] + random.uniform(-0.1, 0.1), direction[1] + random.uniform(-0.1, 0.1))
        if (ballPos[0] + 20 > img.shape[1] and direction[0] > 0) or (ballPos[0] < 20 and direction[0] < 0):
            directX = random.uniform(-0., 0.9)
            direction = (directX, (1 - directX) ** 0.5)
            speed = 3
            ballPos = (img.shape[1] / 2, img.shape[0] / 2)
        if (ballPos[1] + 20 > img.shape[0] and direction[1] > 0) or (ballPos[1] < 20 and direction[1] < 0):
            direction = (direction[0], -direction[1])
        ballPos = (ballPos[0] + direction[0]*speed, ballPos[1] + direction[1]*speed)
    realBallPos = (int(ballPos[0]), int(ballPos[1]))
    cv2.circle(img, realBallPos, 20, (0, 0, 255), 5)
    cv2.circle(img, realBallPos, 10, (255, 0, 0), 5)

    speed *= 1.001
    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
