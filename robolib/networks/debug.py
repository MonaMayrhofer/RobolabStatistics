import matplotlib.pyplot as plt
import numpy as np


def debug_train_data(data, input_size, stride):
    for a, b in data:
        shape = (int(input_size[0]/stride), int(input_size[1]/stride))
        a = np.reshape(a, shape)
        b = np.reshape(b, shape)
        plt.imshow(np.concatenate((a, b)).T, cmap='Greys_r')
        plt.show()


def debug_image(img):
    plt.imshow(img.T, cmap='Greys_r')
    plt.show()
