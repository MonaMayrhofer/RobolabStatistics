import cv2
import numpy as np
import os.path
from urllib.request import urlretrieve
from threading import Thread


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

heatMap = np.zeros((10, 10, 1), dtype=np.uint8)

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
        heatMap.resize(gray.shape)
        oldFaces = faces
        faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, minSize, maxSize, True)
        if len(faces) != 2:
            if timeout > 0:
                timeout -= 1
                faces = oldFaces
            else:
                paused = True
        else:
            print(len(faces))
            if timeout < 10:
                timeout += 1
            paused = False


t = Thread(target=updateCamera)
t.start()

direction = (1., 1.)
speed = 3
ballPos = (100., 100.)

while True:
    if not paused:
        x1, y1, w1, h1 = faces[0]
        x2, y2, w2, h2 = faces[1]
        if(x1 > x2):
            face = faces[0]
            faces[0] = faces[1]
            faces[1] = face
        print(ballPos[0], x1 + w1, x2, ballPos[1], y1 + h1, y1, y2 + h2, y2)
        if (ballPos[0] - 20 < x1 + w1 and ballPos[1] < y1 + h1 and ballPos[1] > y1 and direction[0] < 0)\
                and (ballPos[0] + 20 > x2 and ballPos[1] < y2 + h2 and ballPos[1] > y2 and direction[0] > 0):
            direction = (-direction[0], direction[1])
        if (ballPos[0] + 20 > img.shape[1] and direction[0] > 0) or (ballPos[0] < 20 and direction[0] < 0):
            direction = (-direction[0], direction[1])
        if (ballPos[1] + 20 > img.shape[0] and direction[1] > 0) or (ballPos[1] < 20 and direction[1] < 0):
            direction = (direction[0], -direction[1])
        ballPos = (ballPos[0] + direction[0]*speed, ballPos[1] + direction[1]*speed)
    realBallPos = (int(ballPos[0]), int(ballPos[1]))
    cv2.circle(img, realBallPos, 20, (0, 0, 0), 5)

    cv2.imshow('img', img)

    val, heatThresh = cv2.threshold(heatMap, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow('heatMap', heatThresh)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
