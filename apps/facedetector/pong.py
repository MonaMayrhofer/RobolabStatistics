import cv2
import numpy as np
import os.path
from urllib.request import urlretrieve


def download_xml(url, filename):
    if os.path.isfile(filename):
        print("{} already present, to re-download it remove the file.".format(filename))
    else:
        print("{} not found, downloading it from: {} ".format(filename, url))
        urlretrieve(url, filename)


FRONTALFACE_URL = \
    'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
FRONTALFACE_FILENAME = 'FrontalFace.xml'

download_xml(FRONTALFACE_URL, FRONTALFACE_FILENAME)


face_cascades = cv2.CascadeClassifier(FRONTALFACE_FILENAME)

cap = cv2.VideoCapture(0)

heatMap = np.zeros((10, 10, 1), dtype=np.uint8)

divSpeed = 5
frameCount = 0

minSize = (50, 50)
maxSize = (300, 300)

lastPlayerPos = np.array([(-1, -1), (-1, -1)])


def handleFace(face, level, weight):
    global frameCount
    global heatMap
    x, y, w, h = face
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0))

    cx = int(x + w / 2)
    cy = int(y + h / 2)
    cv2.putText(img, "L{} {}".format(level, weight), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    frameCount += 1
    if frameCount > divSpeed:
        heatMap = cv2.divide(heatMap, 1.1)
        frameCount = 0
    new_mask = np.zeros(heatMap.shape, heatMap.dtype)
    cv2.circle(new_mask, (cx, cy), max(w, h), 32, -1)
    heatMap = cv2.add(heatMap, new_mask)

    return cx, cy

velocity = 3
direction = np.array([1., 1.])
ballPos = np.array([100., 100.])

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heatMap.resize(gray.shape)
    faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, minSize, maxSize, True)
    if(True):
        #playerPos = np.array([handleFace(faces[0], rejectLevels[0], levelWeights[0]),
                              #handleFace(faces[1], rejectLevels[1], levelWeights[1])])
        #cv2.circle(img, tuple(playerPos[0]), 20, (0, 255, 0))
        #cv2.circle(img, tuple(playerPos[1]), 20, (255, 0, 0))
        ballPos = np.polyadd(ballPos, direction*velocity)
        if((ballPos[0] + 10 > img.shape[1] and direction[0] > 0) or (ballPos[0] < 10 and direction[0] < 0)):
            direction = np.array([-direction[0], direction[1]])
        if((ballPos[1] + 10 > img.shape[0] and direction[1] > 0) or (ballPos[1] < 10 and direction[1] < 0)):
            direction = np.array([direction[0], -direction[1]])
        realBallPos = np.array([int(ballPos[0]), int(ballPos[1])])
        print(realBallPos[0], " ", img.shape[1], " ", realBallPos[1], " ", img.shape[0])
        cv2.circle(img, tuple(realBallPos), 20, (0, 0, 0), 5)
        velocity = velocity * 1.005

    cv2.imshow('img', img)

    val, heatThresh = cv2.threshold(heatMap, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow('heatMap', heatThresh)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
