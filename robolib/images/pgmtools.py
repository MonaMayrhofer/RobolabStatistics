import cv2
import numpy as np
import os


def read_pgm(filename):
    assert os.path.exists(filename), "File '{0}' wasn't found".format(filename)
    arr = cv2.imread(filename, 0)
    assert arr is not None, "File '{0}' couldn't be decoded".format(filename)
    arr = np.swapaxes(arr, 0, 1)
    return arr
