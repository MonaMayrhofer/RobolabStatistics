import cv2
import numpy as np


def get_pixel_input(rows, cols, name="Edit Image", dtype=np.float32, low=-1, high=1):
    """Get a small image drawn by the user."""

    def draw_circle(event, x, y, flags, param):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            if event in [cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN] and flags & cv2.EVENT_FLAG_LBUTTON:
                img[y, x] = low if flags & cv2.EVENT_FLAG_SHIFTKEY else high

    img = np.empty((rows, cols, 1), dtype)
    img.fill(low)

    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(name, draw_circle)

    while True:
        cv2.imshow(name, img)
        if wait_for_end_key():
            break
    cv2.destroyAllWindows()

    return np.asarray(img[:, :])


def show_image(mat, name="Image"):
    cv2.namedWindow(name)
    cv2.imshow(name, mat)

    while True:
        if wait_for_end_key():
            break
    cv2.destroyAllWindows()


def wait_for_end_key():
    k = cv2.waitKey(20) & 0xFF
    return k in [27, 13, 32]
