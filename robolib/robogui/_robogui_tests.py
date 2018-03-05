import robolib.robogui.pixel_editor as pe
import numpy as np

size = 20

img = pe.get_drawing_input(9, 9)

print(img.shape)

pe.show_image(img)
