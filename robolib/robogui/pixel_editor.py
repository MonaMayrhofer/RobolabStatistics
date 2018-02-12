import cv2
import numpy as np
from robolib.images.feature_extraction import resize_image_to_info

__DEFAULT_CONTINUE_KEYS = [27, 13, 32]


def get_pixel_input_raw(rows, cols, name="Edit Image", dtype=np.float32, low=-1, high=1, continue_keys=None):
    return np.array(_get_pixel_input_raw(rows, cols, name, dtype, low, high, continue_keys)[:, :])


def _get_pixel_input_raw(rows, cols, name="Edit Image", dtype=np.float32, low=-1, high=1, continue_keys=None):
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

    return img


def get_drawing_input(dst_rows, dst_cols, inp_rows=None, inp_cols=None, name="Input Drawing", dtype=np.float32, low=-1, high=1, continue_keys=None):
    if inp_rows is None:
        inp_rows = dst_rows * 2
    if inp_cols is None:
        inp_cols = dst_cols * 2
    img = _get_pixel_input_raw(inp_rows, inp_cols, name, dtype, low, high, continue_keys)

    img = resize_image_to_info(img, dst_rows, dst_cols, low, high)
    return np.array(img[:, :]).reshape((9, 9, 1))


def show_image(mat, name="Image", end_key=27, continue_keys=None):
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
