import os

import numpy as np
import matplotlib.pyplot as plt
from robolib.images.pgmtools import read_pgm
from robolib.util.random import random_different_numbers
from robolib.networks.debug import debug_image
import random
import warnings

def gen_data_new(train_set_size, class_folder_names, pic_dir, input_image_size=(100, 100), input_to_output_stride=2):
    if input_image_size[0] % input_to_output_stride != 0 and input_image_size[1] % input_to_output_stride != 0:
        raise Exception("Input image size must be divisible by the stride")
    size_1 = int(input_image_size[0]/input_to_output_stride)
    size_2 = int(input_image_size[1]/input_to_output_stride)
    total_image_length = size_1 * size_2
    classes = len(class_folder_names)

    count = 0
    x_tr_positive = np.zeros([train_set_size, 2, total_image_length])  # Save n pairs of images
    y_tr_positive = np.zeros([train_set_size, 1])  # Is this pair a positive or a negative

    # Gen Positive Examples
    for i in range(classes):  # From all classes
        for j in range(int(train_set_size / classes)):  # Get two different images of the same person
            ind1, ind2 = random_different_numbers(10)

            im1 = read_pgm(os.getcwd() + '/'+pic_dir+'/' + class_folder_names[i] + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            im2 = read_pgm(os.getcwd() + '/'+pic_dir+'/' + class_folder_names[i] + '/' + str(ind2 + 1) + '.pgm', 'rw+')

            im1 = im1[::input_to_output_stride, ::input_to_output_stride]
            im2 = im2[::input_to_output_stride, ::input_to_output_stride]

            im1 = im1.reshape(total_image_length)
            im2 = im2.reshape(total_image_length)

            x_tr_positive[count, 0, :] = im1
            x_tr_positive[count, 1, :] = im2
            y_tr_positive[count] = 1
            count += 1

    count = 0
    x_tr_negative = np.zeros([train_set_size, 2, total_image_length])
    y_tr_negative = np.zeros([train_set_size, 1])
    # Gen Negative Examples
    for i in range(int(train_set_size / 10)):  # Für ein Zehntel der Testdaten
        for j in range(10):  # Jeweils 10 Bilder auswählen
            ind1, ind2 = random_different_numbers(classes)

            im1 = read_pgm(os.getcwd() + '/'+pic_dir+'/' + class_folder_names[ind1] + '/' + str(j + 1) + '.pgm', 'rw+')
            im2 = read_pgm(os.getcwd() + '/'+pic_dir+'/' + class_folder_names[ind2] + '/' + str(j + 1) + '.pgm', 'rw+')

            im1 = im1[::input_to_output_stride, ::input_to_output_stride]
            im2 = im2[::input_to_output_stride, ::input_to_output_stride]

            im1 = im1.reshape(im1.shape[0] * im1.shape[1])
            im2 = im2.reshape(im2.shape[0] * im2.shape[1])

            x_tr_negative[count, 0, :] = im1
            x_tr_negative[count, 1, :] = im2
            y_tr_negative[count] = 0
            count += 1

    x_train = np.concatenate([x_tr_positive, x_tr_negative], axis=0) / 255  # Squish training-data from 0-255 to 0-1
    y_train = np.concatenate([y_tr_positive, y_tr_negative], axis=0)

    return x_train, y_train


def load_one_image(referenceimgpath, name, img, show=False, amount=5):
    if img is not None:
        img = read_pgm(referenceimgpath + "/" + name + "/" + str(img) + ".pgm")
        if show:
            debug_image(img)
        return img
    imgs = []
    selection = os.listdir(os.path.join(referenceimgpath, name))
    selection = random.sample(selection, min(len(selection), amount))
    for file in selection:
        img = read_pgm(os.path.join(referenceimgpath, name, file))
        imgs.append(img)
    if show:
        warnings.warn("Show is not yet supported with bulk-load", RuntimeWarning)  # TODO Show with bulk-load
    return imgs

