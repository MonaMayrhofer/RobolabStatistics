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
EYE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml'
EYE_FILENAME = 'Eye.xml'

download_xml(FRONTALFACE_URL, FRONTALFACE_FILENAME)
download_xml(EYE_URL, EYE_FILENAME)


face_cascades = cv2.CascadeClassifier(FRONTALFACE_FILENAME)
eye_cascades = cv2.CascadeClassifier(EYE_FILENAME)

cap = cv2.VideoCapture(0)

heatMap = np.zeros((10, 10, 1), dtype=np.uint8)

divSpeed = 5
frameCount = 0

minSize = (50, 50)
maxSize = (300, 300)

lastPlayerPos = np.array([(-1, -1), (-1, -1)])

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heatMap.resize(gray.shape)
    faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, minSize, maxSize, True)

    highestInd = -1
    secondhighestInd = -1

    for index, val in enumerate(levelWeights):
        if highestInd == -1 or val > levelWeights[highestInd]:
            highestInd = index
        elif secondhighestInd == -1 or val > levelWeights[secondhighestInd]:
            secondhighestInd = index

    def handleFace(face, level, weight):
        global frameCount
        global heatMap
        x, y, w, h = face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0))

        cx = int(x+w/2)
        cy = int(y+h/2)
        cv2.putText(img, "L{} {}".format(level, weight), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        frameCount += 1
        if frameCount > divSpeed:
            heatMap = cv2.divide(heatMap, 1.1)
            frameCount = 0
        new_mask = np.zeros(heatMap.shape, heatMap.dtype)
        cv2.circle(new_mask, (cx, cy), max(w, h), 32, -1)
        heatMap = cv2.add(heatMap, new_mask)

        return cx, cy

    playerPos = np.array([(-1, -1), (-1, -1)])

    if highestInd >= 0:
        playerPos[0] = handleFace(faces[highestInd], rejectLevels[highestInd], levelWeights[highestInd])
    if secondhighestInd >= 0:
        playerPos[1] = handleFace(faces[secondhighestInd], rejectLevels[secondhighestInd], levelWeights[secondhighestInd])

    # Create Scores
    if highestInd >= 0 and secondhighestInd >= 0:
        currentPlayerPos = np.array([(-1, -1), (-1, -1)])

        abscore = np.linalg.norm(playerPos[0] - lastPlayerPos[0]) + np.linalg.norm(playerPos[1] - lastPlayerPos[1])
        bascore = np.linalg.norm(playerPos[1] - lastPlayerPos[0]) + np.linalg.norm(playerPos[0] - lastPlayerPos[1])

        indexA = 0
        indexB = 1
        if bascore < abscore:
            indexA = 1
            indexB = 0

        currentPlayerPos[indexA] = playerPos[0]
        currentPlayerPos[indexB] = playerPos[1]

        print(currentPlayerPos[0])
        print(lastPlayerPos[0])
        cv2.line(img, tuple(lastPlayerPos[0]), tuple(currentPlayerPos[0]), (0, 255, 0))
        cv2.line(img, tuple(lastPlayerPos[1]), tuple(currentPlayerPos[1]), (255, 0, 0))

        lastPlayerPos = currentPlayerPos

    cv2.circle(img, tuple(lastPlayerPos[0]), 20, (0, 255, 0))
    cv2.circle(img, tuple(lastPlayerPos[1]), 20, (255, 0, 0))

    cv2.imshow('img', img)

    val, heatThresh = cv2.threshold(heatMap, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow('heatMap', heatThresh)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
