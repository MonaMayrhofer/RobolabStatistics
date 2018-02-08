import cv2
import numpy as np


def get_pixel_input(rows, cols, name="Edit Image"):
    """Get a small image drawn by the user."""

    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            img[y, x] = 0 if flags & cv2.EVENT_FLAG_SHIFTKEY else 1

    img = np.zeros((rows, cols, 1), np.float32)
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(name, draw_circle)

    while True:
        cv2.imshow(name, img)
        k = cv2.waitKey(20) & 0xFF
        if k in [27, 13, 32]:
            break
    cv2.destroyAllWindows()

    return np.asarray(img[:, :])
