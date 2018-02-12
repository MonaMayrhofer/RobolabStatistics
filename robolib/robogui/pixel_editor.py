import cv2
import numpy as np

__DEFAULT_CONTINUE_KEYS = [27, 13, 32]


def get_pixel_input(rows, cols, name="Edit Image", dtype=np.float32, low=-1, high=1, continue_keys=None):
    """Get a small image drawn by the user."""

    if continue_keys is None:
        continue_keys = __DEFAULT_CONTINUE_KEYS

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
        if wait_for_end_key(continue_keys):
            break
    cv2.destroyAllWindows()

    return np.asarray(img[:, :])


def show_image(mat, name="Image", end_key=27 , continue_keys=None):
    if continue_keys is None:
        continue_keys = __DEFAULT_CONTINUE_KEYS

    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, mat)

    ret = False

    while True:
        k = wait_for_end_key(continue_keys)
        if k:
            if k == end_key:
                ret = True
            break
    cv2.destroyAllWindows()

    return ret


def wait_for_end_key(continue_keys):
    k = cv2.waitKey(20) & 0xFF
    return k if k in continue_keys else 0
