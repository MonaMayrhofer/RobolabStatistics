import keras.backend as K
import numpy as np
import os
import re
import robolib.datamanager.atntfaces as data


def get_data():
    data.get_data("AtnTFaces", True)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


def gen_data_new(samp_f, total_to_samp, pic_dir="ModelData_AtnTFaces", classes=4):
    # first run on 1 data to find array shape
    im1 = read_pgm(os.getcwd() + '/'+pic_dir+'/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    im1 = im1[::samp_f, ::samp_f]
    sz_1 = im1.shape[0]
    sz_2 = im1.shape[1]

    count = 0
    x_tr_m = np.zeros([total_to_samp, 2, sz_2 * sz_1])
    y_tr_m = np.zeros([total_to_samp, 1])
    for i in range(classes):
        for j in range(int(total_to_samp / classes)):
            # let's make the pairs different, same one is adding no value
            ind1 = 0
            ind2 = 0
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)

            im1 = read_pgm(os.getcwd() + '/'+pic_dir+'/s' + str(i + 1) + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            im2 = read_pgm(os.getcwd() + '/'+pic_dir+'/s' + str(i + 1) + '/' + str(ind2 + 1) + '.pgm', 'rw+')

            im1 = im1[::samp_f, ::samp_f]
            im2 = im2[::samp_f, ::samp_f]

            x_tr_m[count, 0, :] = im1.reshape(im1.shape[0] * im1.shape[1])
            x_tr_m[count, 1, :] = im2.reshape(im1.shape[0] * im1.shape[1])
            y_tr_m[count] = 1
            count += 1

            # plt.figure(1)
            # plt.imshow(im1, cmap='Greys_r')
            # plt.figure(2)
            # plt.imshow(im2, cmap='Greys_r')
            # plt.show()

    count = 0
    x_tr_non = np.zeros([total_to_samp, 2, sz_2 * sz_1])
    y_tr_non = np.zeros([total_to_samp, 1])
    for i in range(int(total_to_samp / 10)):
        for j in range(10):
            while True:
                ind1 = np.random.randint(classes)
                ind2 = np.random.randint(classes)
                if ind1 != ind2:
                    break

            im1 = read_pgm(os.getcwd() + '/'+pic_dir+'/s' + str(ind1 + 1) + '/' + str(j + 1) + '.pgm', 'rw+')
            im2 = read_pgm(os.getcwd() + '/'+pic_dir+'/s' + str(ind2 + 1) + '/' + str(j + 1) + '.pgm', 'rw+')

            im1 = im1[::samp_f, ::samp_f]
            im2 = im2[::samp_f, ::samp_f]

            x_tr_non[count, 0, :] = im1.reshape(im1.shape[0] * im1.shape[1])
            x_tr_non[count, 1, :] = im2.reshape(im1.shape[0] * im1.shape[1])
            y_tr_non[count] = 0
            count += 1

            # plt.figure(1)
            # plt.imshow(im1, cmap='Greys_r')
            # plt.figure(2)
            # plt.imshow(im2, cmap='Greys_r')
            # plt.show()

    x_train = np.concatenate([x_tr_m, x_tr_non], axis=0) / 255
    y_train = np.concatenate([y_tr_m, y_tr_non], axis=0)

    return x_train.astype('float32'), y_train.astype('float32')
