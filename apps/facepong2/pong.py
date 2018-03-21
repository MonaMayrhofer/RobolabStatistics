import pygame
import pygame.locals
import cv2
import os
import numpy as np
import sys
# from apps.facepong import camOpener
import random
import time
import pymunk
from apps.facepong2.pongGame import PongGame
from apps.facepong2.pongRenderer import PongRenderer
"""
#  ==PYGAME==


# ==Win==
pointsToWin = camOpener.get_wins()


# ==GameStats==
frameCount = 0
paused = True
timeout = 10
lastFaces = [(0, 0), (0, 0)]

# ==Ball Stats==
directY = random.uniform(-0.9, 0.9)
direction = ((1 - directY) ** 0.5, directY)
speed = 300  # Pixel/Sec
ballPos = (img.shape[1] / 2, img.shape[0] / 2)

# ==FPS==
lastLoop = time.time()

# ==MODEL==

# == Pymunk ==
insets = (80, 20)  # Top, Bottom

pymunkSpace = pymunk.Space()
pymunkSpace.gravity = (0.0, 0.0)

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
faceOneShape.elasticity = 0.8
faceTwoBody = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
faceTwoShape = pymunk.Circle(faceTwoBody, 50, (0, 0))
faceTwoShape.elasticity = 0.8

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
bottomBody.position = 0, height-insets[1]
pymunkSpace.add(bottomBody, bottomShape)

topBody = pymunk.Body(body_type=pymunk.Body.STATIC)
topShape = pymunk.Poly(topBody, [(0, -borderThickness), (0, 0), (width, 0), (width, -borderThickness)])
topShape.elasticity = 1.0
topBody.position = 0, insets[0]
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

debug = np.zeros(img.shape)

winTime = 0
shouldDebug = True


def min_size():
    return (30, 30)


def max_size():
    return (500, 500)


def max_speed():
    return 30


def resize(l_tuple, l_new_len):
    length = (l_tuple[0]**2+l_tuple[1]**2)**0.5
    if length > l_new_len:
        normal = (l_tuple[0]/length*l_new_len, l_tuple[1]/length*l_new_len)
    else:
        normal = l_tuple
    return normal


def reset():
    global speed
    global ballPos
    speed = 300
    ballBody.position = (width / 2, height / 2)
    ballPos = ballBody.position
    l_dir = random.randint(0, 1)
    if l_dir == 0:
        ballBody.velocity = (50, 0)
    else:
        ballBody.velocity = (-50, 0)


def win():
    r = random.uniform(0, 9999999)
    d = os.path.dirname(__file__)
    filename = os.path.join(d, '/winFaces/Test0.png')
    print(filename)
    # cv2.imwrite(filename, img)


reset()

# === MAIN LOOP ===

winPaused = False


cv2.destroyAllWindows()
"""

PongGame().run()

print("Goodbye!")
