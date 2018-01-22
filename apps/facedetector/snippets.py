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
maxSize = (300, 300)

ret, img = cap.read()
paused = True
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, minSize, maxSize, True)

directY = random.uniform(-0.9, 0.9)
direction = ((1-directY) ** 0.5, directY)
speed = 3
ballPos = (img.shape[1]/2, img.shape[0]/2)

timeout = 10
while True:
    ret, img = cap.read()
    cv2.flip(img,1,img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    oldFaces = faces

    fieldsize = int(img.shape[1] / 3)
    # faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, minSize, maxSize, True)

    facesLeft, rejectLevelsLeft, levelWeightsLeft = face_cascades.detectMultiScale3(gray[0:img.shape[0], 0:fieldsize], 1.3, 5, 0, minSize, maxSize, True)
    facesRight, rejectLevelsRight, levelWeightsRight = face_cascades.detectMultiScale3(gray[0:img.shape[0], 2*fieldsize:3*fieldsize], 1.3, 5, 0, minSize, maxSize, True)

    # TODO Wichtigstes Gesicht ermitteln


    if len(facesLeft) != 0 and len(facesRight) != 0:
        paused = False
        timeout = 10
        leftInd = np.argmax(levelWeightsLeft)
        rightInd = np.argmax(levelWeightsRight)

        faces = [facesLeft[leftInd], facesRight[rightInd]]
        faces[1][0] += 2*fieldsize
    else:
        if timeout > 0:
            timeout -= 1
        else:
            paused = True

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Game Data
    if not paused:
        x1, y1, w1, h1 = faces[0]
        x2, y2, w2, h2 = faces[1]
        z1 = x1 + w1
        t1 = y1 + h1
        z2 = x2 + w2
        t2 = y2 + h2
        if (ballPos[0] - 20 < z1 and t1 > ballPos[1] > y1 and direction[0] < 0 and ballPos[
            0] + w1 / 2 > z1) \
                or (ballPos[0] + 20 > x2 and t2 > ballPos[1] > y2 and direction[0] > 0 \
                    and ballPos[0] - w2/ 2 < z2):
            direction = (-direction[0] + random.uniform(-0.1, 0.1), direction[1] + random.uniform(-0.1, 0.1))
        if (ballPos[0] + 20 > img.shape[1] and direction[0] > 0) or (ballPos[0] < 20 and direction[0] < 0):
            directX = random.uniform(-0.9, 0.9)
            direction = (directX, (1 - directX) ** 0.5)
            speed = 3
            ballPos = (img.shape[1] / 2, img.shape[0] / 2)
        if (ballPos[1] + 20 > img.shape[0] and direction[1] > 0) or (ballPos[1] < 20 and direction[1] < 0):
            direction = (direction[0], -direction[1])
        ballPos = (ballPos[0] + direction[0] * speed, ballPos[1] + direction[1] * speed)
        if speed < 20:
            speed *= 1.005
    realBallPos = (int(ballPos[0]), int(ballPos[1]))
    cv2.circle(img, realBallPos, 20, (0, 0, 255), 5)
    cv2.circle(img, realBallPos, 10, (255, 0, 0), 5)
    cv2.line(img, (int(img.shape[1] / 3), 0), (int(img.shape[1] / 3), img.shape[0]), (0, 0, 0), 2)
    cv2.line(img, (int(img.shape[1] / 3 * 2), 0), (int(img.shape[1] / 3 * 2), img.shape[0]), (0, 0, 0), 2)

    textPos = int(img.shape[1] / 2) - 100
    cv2.putText(img, "Paused: {}".format(paused), (textPos, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Timeout: {}".format(timeout), (textPos, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Speed: {}".format(speed), (textPos, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
