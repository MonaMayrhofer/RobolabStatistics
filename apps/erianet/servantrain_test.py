from robolib.networks.erianet import Erianet
import matplotlib.pyplot as plt
import cv2
import numpy as np

insets = (12, q5, 12, 8)
# size = (64-insets[1]-insets[3], 64-insets[0]-insets[2])
net = Erianet("3BHIF", input_image_size=(128, 128), input_to_output_stride=2, insets=insets)
size = net.get_input_dim_for((128, 128), 2, insets, 2)
x, y = net.get_train_data(100, "3BHIF", None)

for (a, b), l in zip(x, y):
    plt.figure(1)
    a = np.reshape(a, size)
    b = np.reshape(b, size)
    img = cv2.hconcat([a, b])
    plt.imshow(img, cmap='Greys_r', )
    plt.text(0, -5, str(l))
    plt.show()
