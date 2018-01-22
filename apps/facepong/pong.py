import random

import cv2
import numpy as np
import time

import pymunk

import robolib.modelmanager.downloader as downloader

# ==MODEL==
MODEL_FILE = 'FrontalFace.xml'
downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_DEFAULT, MODEL_FILE)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)

# ==WINDOW==
WINDOW_NAME = 'img'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, 1000, 800)
fullscreen = False

# ==OPENCV==
cap = cv2.VideoCapture(0)
_, img = cap.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = []
minSize = (50, 50)
# maxSize = (150, 150)
maxSize = (300, 300)

# ==GameStats==
frameCount = 0
paused = True
timeout = 10

# ==Ball Stats==
directY = random.uniform(-0.9, 0.9)
direction = ((1 - directY) ** 0.5, directY)
speed = 30  # Pixel/Sec
ballPos = (img.shape[1] / 2, img.shape[0] / 2)

# ==FPS==
lastLoop = time.time()


# == Pymunk ==
pymunkSpace = pymunk.Space()
pymunkSpace.gravity = (0.0, 0.0)
pymunkSpace.damping = 0

ballBody = pymunk.Body(10, 25)
ballShape = pymunk.Circle(ballBody, 20, (0, 0))

# ballBody.position = (img.shape[1] / 2, img.shape[0] / 2)

ballBody.position = 100, 100

pymunkSpace.add(ballBody, ballShape)

faceOneBody = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
faceOneShape = pymunk.Circle(faceOneBody, 20, (0, 0))
faceOneShape.elasticity = 1.0
faceTwoBody = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
faceTwoShape = pymunk.Circle(faceTwoBody, 20, (0, 0))
faceTwoShape.elasticity = 1.0

faceOneBody.position = (0, 0)
faceTwoBody.position = (0, 0)

pymunkSpace.add(faceOneBody, faceOneShape)
pymunkSpace.add(faceTwoBody, faceTwoShape)

static_lines = [pymunk.Segment(pymunkSpace.static_body, (0, 0), (0, img.shape[0]), 2),
                pymunk.Segment(pymunkSpace.static_body, (0, img.shape[0]), (img.shape[1], img.shape[0]), 2),
                pymunk.Segment(pymunkSpace.static_body, (img.shape[1], img.shape[0]), (img.shape[1], 0), 2),
                pymunk.Segment(pymunkSpace.static_body, (img.shape[1], 0), (0, 0), 2)]
for line in static_lines:
    line.elasticity = 1.0

pymunkSpace.add(static_lines)

while True:
    # == Calc FPS
    currentTime = time.time()
    delta = currentTime-lastLoop
    lastLoop = currentTime
    fps = 1/delta

    # == Read Image ==
    _, img = cap.read()
    cv2.flip(img, 1, img)
    debug = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    field_size = int(img.shape[1] / 3)

    facesLeft, rejectLevelsLeft, levelWeightsLeft = face_cascades.detectMultiScale3(gray[0:img.shape[0], 0:field_size],
                                                                                    1.3, 5, 0, minSize, maxSize, True)
    facesRight, rejectLevelsRight, levelWeightsRight = face_cascades.detectMultiScale3(
        gray[0:img.shape[0], 2 * field_size:3 * field_size], 1.3, 5, 0, minSize, maxSize, True)

    if len(facesLeft) != 0 and len(facesRight) != 0:
        paused = False
        timeout = 10
        leftInd = np.argmax(levelWeightsLeft)
        rightInd = np.argmax(levelWeightsRight)

        faces = [facesLeft[leftInd], facesRight[rightInd]]
        faces[1][0] += 2 * field_size
    else:
        if len(facesLeft) == 0 and timeout == 0:
            cv2.rectangle(img, (0, 0), (field_size, img.shape[0]), (0, 0, 255), 5)
        if len(facesRight) == 0 and timeout == 0:
            cv2.rectangle(img, (2 * field_size, 0), (3 * field_size, img.shape[0]), (0, 0, 255), 5)
        if timeout > 0:
            timeout -= 1
        else:
            paused = True

    # == Show detected faces ==
    for (x, y, w, h) in facesLeft:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in facesRight:
        cv2.rectangle(img, (x+2 * field_size, y), (x + w + 2 * field_size, y + h), (0, 255, 0), 2)

    # == Game Loop ==
    if not paused:
        if delta != 0:
            pymunkSpace.step(delta)

        x1, y1, w1, h1 = faces[0]
        x2, y2, w2, h2 = faces[1]

        # faceOneBody.position = (x1+w1/2, y1+h1/2)
        # faceTwoBody.position = (x2+w2/2, y2+h2/2)
        '''
        z1, z2 = x1 + w1, x2 + w2
        t1, t2 = y1 + h1, y2 + h2
        # Collision detection [Faces]
        if (ballPos[0] - 20 < z1 and t1 > ballPos[1] > y1 and direction[0] < 0 and ballPos[0] + w1 / 2 > z1) \
                or (ballPos[0] + 20 > x2 and t2 > ballPos[1] > y2 and direction[0] > 0 and ballPos[0] - w2 / 2 < z2):
            direction = (-direction[0] + random.uniform(-0.1, 0.1), direction[1] + random.uniform(-0.1, 0.1))

        # Goal collision
        if (ballPos[0] + 20 > img.shape[1] and direction[0] > 0) or (ballPos[0] < 20 and direction[0] < 0):
            directX = random.uniform(-0.9, 0.9)
            direction = (directX, (1 - directX) ** 0.5)
            speed = 30
            ballPos = (img.shape[1] / 2, img.shape[0] / 2)

        # Border collision
        if (ballPos[1] + 20 > img.shape[0] and direction[1] > 0) or (ballPos[1] < 20 and direction[1] < 0):
            direction = (direction[0], -direction[1])
        '''
        # Move ball
        # ballPos = (ballPos[0] + direction[0] * speed * delta, ballPos[1] + direction[1] * speed * delta)
        ballPos = ballBody.position
        print(ballPos)
        # Speed increase
        if speed < 300:
            speed *= 1.005

    ballPos = ballBody.position

    # == Draw Ball ==
    realBallPos = (int(ballPos[0]), int(ballPos[1]))
    cv2.circle(img, realBallPos, 20, (0, 0, 255), 5)
    cv2.circle(img, realBallPos, 10, (255, 0, 0), 5)

    cv2.circle(img, (int(faceOneBody.position.x), int(faceOneBody.position.y)), 20, (255, 0, 0), 3)
    cv2.circle(img, (int(faceTwoBody.position.x), int(faceTwoBody.position.y)), 20, (255, 0, 0), 3)

    # == Draw Fieldlines ==
    cv2.line(img, (int(img.shape[1] / 3), 0), (int(img.shape[1] / 3), img.shape[0]), (0, 0, 0), 2)
    cv2.line(img, (int(img.shape[1] / 3 * 2), 0), (int(img.shape[1] / 3 * 2), img.shape[0]), (0, 0, 0), 2)

    # == Debug Data ==
    textPos = int(img.shape[1] / 2) - 100
    cv2.putText(debug, "Paused: {}".format(paused), (textPos, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(debug, "Timeout: {}".format(timeout), (textPos, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(debug, "Speed: {}".format(speed), (textPos, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(debug, "FPS: {:.2f}".format(fps), (textPos, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(WINDOW_NAME, img)
    cv2.imshow('debugwindow', debug)

    # == Key-Controls ==
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == 200:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if fullscreen else cv2.WINDOW_FULLSCREEN)
        fullscreen = not fullscreen

cap.release()
cv2.destroyAllWindows()
