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
lastFaces = [(0, 0), (0, 0)]

# ==Ball Stats==
directY = random.uniform(-0.9, 0.9)
direction = ((1 - directY) ** 0.5, directY)
speed = 30  # Pixel/Sec
ballPos = (img.shape[1] / 2, img.shape[0] / 2)

# ==FPS==
lastLoop = time.time()


def reset():
    ballBody.position = (width / 2, height / 2)
    dir = random.randint(0, 1)
    if(dir == 0):
        ballBody.velocity = (50, 0)
    else:
        ballBody.velocity = (-50, 0)


# == Pymunk ==
pymunkSpace = pymunk.Space()
pymunkSpace.gravity = (0.0, 0.0)
pymunkSpace.damping = 0.8

mass = 10
radius = 25
inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
ballBody = pymunk.Body(mass, inertia)
ballShape = pymunk.Circle(ballBody, radius, (0, 0))
ballShape.elasticity = 0.95
ballShape.friction = 0.9

pymunkSpace.add(ballBody, ballShape)

faceOneBody = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
faceOneShape = pymunk.Circle(faceOneBody, 50, (0, 0))
faceOneShape.elasticity = 1.0
faceTwoBody = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
faceTwoShape = pymunk.Circle(faceTwoBody, 50, (0, 0))
faceTwoShape.elasticity = 1.0

faceOneBody.position = (0, 0)
faceTwoBody.position = (0, 0)

width = img.shape[1]
height = img.shape[0]

pymunkSpace.add(faceOneBody, faceOneShape)
pymunkSpace.add(faceTwoBody, faceTwoShape)

borderThickness = 100

bottomBody = pymunk.Body(body_type=pymunk.Body.STATIC)
bottomShape = pymunk.Poly(bottomBody, [(0, 0), (0, borderThickness), (width, borderThickness), (width, 0)])
bottomShape.elasticity = 1.0
bottomBody.position = 0, height
pymunkSpace.add(bottomBody, bottomShape)

topBody = pymunk.Body(body_type=pymunk.Body.STATIC)
topShape = pymunk.Poly(topBody, [(0, -borderThickness), (0, 0), (width, 0), (width, -borderThickness)])
topShape.elasticity = 1.0
topBody.position = 0, 0
pymunkSpace.add(topBody, topShape)

leftBody = pymunk.Body(body_type=pymunk.Body.STATIC)
leftShape = pymunk.Poly(leftBody, [(-borderThickness, -borderThickness), (-borderThickness, height+borderThickness),
                                   (0, height+borderThickness), (0, -borderThickness)])
leftShape.elasticity = 1.0
leftBody.position = 0, 0
pymunkSpace.add(leftBody, leftShape)

rightBody = pymunk.Body(body_type=pymunk.Body.STATIC)
rightShape = pymunk.Poly(rightBody, [(0, -borderThickness), (0, height+borderThickness),
                                     (borderThickness, height+borderThickness), (borderThickness, -borderThickness)])
rightShape.elasticity = 1.0
rightBody.position = width, 0
pymunkSpace.add(rightBody, rightShape)

slowdown = 1

pointsLeft = 0
pointsRight = 0

reset()

while True:
    # == Calc FPS
    currentTime = time.time()
    delta = currentTime-lastLoop
    lastLoop = currentTime
    fps = 1/delta

    # == Read Image ==
    _, img = cap.read()
    debug = np.zeros(img.shape)
    cv2.flip(img, 1, img)
    debug = np.zeros(img.shape)
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
        cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in facesRight:
        cv2.rectangle(debug, (x+2 * field_size, y), (x + w + 2 * field_size, y + h), (0, 255, 0), 2)

    # == Game Loop ==
    if not paused:

        x1, y1, w1, h1 = faces[0]
        x2, y2, w2, h2 = faces[1]

        currFaces = [(x1 + w1 / 2, y1 + h1 / 2), (x2 + w2 / 2, y2 + h2 / 2)]
        faceVelocities = np.divide(np.subtract(currFaces, lastFaces), max(delta, 0.00001))
        lastFaces = currFaces

        faceOneBody.velocity = faceVelocities[0]*slowdown
        faceTwoBody.velocity = faceVelocities[1]*slowdown

        if delta != 0:
            pymunkSpace.step(delta/slowdown)

        # Move ball
        ballPos = ballBody.position

        # Detect goals
        if ballPos[0] < 25:
            # RESET
            pointsRight += 1
            reset()
        elif ballPos[0] + 25 > width:
            # RESET
            pointsLeft += 1
            reset()

        # Speed increase
        if speed < 300:
            speed *= 1.005

    # == Draw Ball ==
    realBallPos = (int(ballPos[0]), int(ballPos[1]))
    cv2.circle(img, realBallPos, 20, (0, 0, 255), 5)
    cv2.circle(img, realBallPos, 10, (255, 0, 0), 5)

    cv2.circle(img, (int(faceOneBody.position.x), int(faceOneBody.position.y)), 50, (255, 0, 0), 3)
    cv2.circle(img, (int(faceTwoBody.position.x), int(faceTwoBody.position.y)), 50, (255, 0, 0), 3)

    # == Draw Fieldlines ==
    cv2.line(img, (int(img.shape[1] / 3), 0), (int(img.shape[1] / 3), img.shape[0]), (0, 0, 0), 2)
    cv2.line(img, (int(img.shape[1] / 3 * 2), 0), (int(img.shape[1] / 3 * 2), img.shape[0]), (0, 0, 0), 2)

    # == Debug Data ==
    textPos = int(img.shape[1] / 2) - 100
    cv2.putText(debug, "Paused: {}".format(paused), (textPos, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(debug, "Timeout: {}".format(timeout), (textPos, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(debug, "Speed: {}".format(speed), (textPos, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(debug, "FPS: {:.2f}".format(fps), (textPos, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # == Show Points
    pointsPos = textPos + 75
    if pointsLeft > 9:
        pointsPos -= 25
    cv2.putText(img, "{}:{}".format(pointsLeft, pointsRight), (pointsPos, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # == Update Windows ==
    cv2.imshow(WINDOW_NAME, img)
    cv2.imshow('debugwindow', debug)

    # == Key-Controls ==
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == 114:
        reset()
    elif k == 200:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if fullscreen else cv2.WINDOW_FULLSCREEN)
        fullscreen = not fullscreen
    elif k == 114:
        reset()

cap.release()
cv2.destroyAllWindows()
