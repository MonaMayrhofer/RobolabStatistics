import cv2
import numpy as np


def read_pgm(filename):
    print("READ")
    arr = cv2.imread(filename, 0)
    arr = np.swapaxes(arr, 0, 1)
    return arr
