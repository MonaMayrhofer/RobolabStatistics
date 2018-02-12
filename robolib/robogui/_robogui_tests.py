import numpy as np
import cv2
import robolib.robogui.pixel_editor as pe

size = 20

while True:
    img = np.zeros((size, size))
    lineZ = np.random.randint(0, size)
    endLineZ = np.clip(lineZ + np.random.randint(-1, 2), 0, size)

    cv2.line(img, (0, lineZ), (size, endLineZ), 1.0)

    if pe.show_image(img):
        break
