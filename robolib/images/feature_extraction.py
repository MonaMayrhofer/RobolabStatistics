import cv2
from robolib.util.lists import reverse_enumerate
from robolib.util.lists import enumerate_columns
from robolib.util.lists import reverse_enumerate_columns


def crop_image_to_info(mat, low=-1):

    minRow = 0
    for i, row in enumerate(mat):
        if any(pix != low for pix in row):
            minRow = i
            break

    maxRow = 0
    for i, row in reverse_enumerate(mat):
        if any(pix != low for pix in row):
            maxRow = i
            break

    minCol = 0
    for i, col in enumerate_columns(mat):
        print(col)
        if any(pix != low for pix in col):
            minCol = i
            break

    maxCol = 0
    for i, col in reverse_enumerate_columns(mat):
        if any(pix != low for pix in col):
            maxCol = i
            break

    return mat[minRow:maxRow+1, minCol:maxCol+1]
