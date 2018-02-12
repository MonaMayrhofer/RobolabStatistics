import numpy as np
import keras
import robolib.robogui.pixel_editor as pe
import cv2

DEBUG = True

label_labels = ["X", "O"]
labels = np.random.randint(0, 2, size=(1000, 1))
size = 9
data = np.zeros(shape=(1000, size, size, 1))

for la, d in zip(labels, data):
    img = np.empty((size, size))
    img.fill(-1)

    if la == 0:
        cv2.ellipse(img, (4, 4), (np.random.randint(1, 5), np.random.randint(1, 5)), 0, 360, 0, 1)
    else:
        randPointStart = np.random.randint(0, 16)
        randPointEnd = np.random.randint(0, 16)
        cv2.line(img, (int(randPointStart / 4), randPointStart % 4), (8 - int(randPointEnd / 4), 8 - randPointEnd % 4), 1)
        randPointStart = np.random.randint(0, 16)
        randPointEnd = np.random.randint(0, 16)
        cv2.line(img, (8 - int(randPointStart / 4), randPointStart % 4), (int(randPointEnd / 4), 8 - randPointEnd % 4), 1)

    d[:, :, :] = np.reshape(img, (size, size, 1))

    if DEBUG:
        pe.show_image(img)
