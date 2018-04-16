import os
from os import path
import numpy as np
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Lambda, Conv2D, Flatten
from robolib.datamanager.siamese_data_loader import load_one_image
from robolib.networks.common import contrastive_loss, euclidean_dist_output_shape, euclidean_distance
from robolib.util.random import random_different_numbers
from keras import backend
from robolib.images.pgmtools import read_pgm
from robolib.util.decorations import deprecated


class Erianet:
    def __init__(self, model_path, input_image_size=(128, 128), insets=(0, 0, 0, 0), input_to_output_stride=2, do_not_init=False):
        self.input_image_size = input_image_size
        self.input_to_output_stride = input_to_output_stride
        self.model = None
        self.insets = np.asarray(insets)
        self.input_dim = self.get_input_dim_for(input_image_size, input_to_output_stride, self.insets, 1)
        if not do_not_init:
            if model_path is None or not path.isfile(model_path):
                self.create(input_image_size, input_to_output_stride)
            else:
                self.load(model_path)

    @staticmethod
    def get_input_dim_for(input_image_size, input_to_output_stride, insets, dims):
        if dims == 1:
            return ((int(input_image_size[0] / input_to_output_stride)-insets[1]-insets[3]) *
                    (int(input_image_size[1] / input_to_output_stride)-insets[0]-insets[2]), )
        elif dims == 2:
            return (int(input_image_size[0] / input_to_output_stride)-insets[1]-insets[3],
                    int(input_image_size[1] / input_to_output_stride)-insets[0]-insets[2], 1)

    def train(self, data_folder, epochs=100, data_selection=None, callbacks=None, test_percent=0):
        if callbacks is None:
            callbacks = []

        # TODO GEN_DATA_NEW TO HERE
        x, y = self.get_train_data(1000, data_folder, data_selection)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_percent)

        self.model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25, batch_size=128, verbose=2,
                       epochs=epochs,
                       callbacks=callbacks)

    def get_train_data(self, amount, data_folder, data_selection=None):
        if data_selection is None:
            data_selection = self.__get_names_of(data_folder)

        # TODO GEN_DATA_NEW TO HERE
        return self.gen_data_new(amount, data_selection, data_folder, self.input_image_size, self.input_to_output_stride)

    def create(self, input_image_size=(128, 128), input_to_output_stride=2):
        assert all(np.mod(input_image_size, input_to_output_stride) == (0, 0))
        self.model = self.create_erianet()

        rms = RMSprop()
        self.model.compile(loss=contrastive_loss, optimizer=rms)

    def save(self, modelpath):
        print("Saving model to {}".format(modelpath))
        self.model.save(modelpath)

    def load(self, modelpath):
        print("Loading model from File {}".format(modelpath))
        self.model = load_model(modelpath, custom_objects={'contrastive_loss': contrastive_loss, 'backend': backend})

    def predict(self, input_img, reference_data_path, candidates=None, give_all=False):

        input_img = self.preprocess(input_img)

        if candidates is None:
            candidates = self.__get_names_of(reference_data_path)
        probabilities = np.array([], dtype=[('class', int), ('probability', float)])
        for i in range(0, len(candidates)):
            # TODO  REF IMAGE INDEX FOR
            reference_img = self.load_image(reference_data_path, candidates[i], 1, False, preprocess=True)
            probability = float(self.model.predict([input_img, reference_img]))
            pair = (i, probability)
            probabilities = np.append(probabilities, np.array(pair, dtype=probabilities.dtype))
        probabilities = np.sort(probabilities, order='probability')
        probs = probabilities
        certainties = []
        biggestind = 0
        for i in range(len(probs)):
            if i != len(probs)-1:
                certainty = probs[i + 1][1] - probs[i][1]
            else:
                certainty = 0
            certainties.append([candidates[probs[i][0]], probs[i][0], probs[i][1], certainty])
            if certainties[biggestind][2] < certainty:
                biggestind = i
        if give_all:
            return certainties
        return certainties[0:biggestind + 1]

    @staticmethod
    def __get_names_of(folder):
        return next(os.walk(folder))[1]

    def load_image(self, reference_path, name, img, show=False, stride=None, preprocess=False):
        if stride is None:
            stride = self.input_to_output_stride
        image = load_one_image(reference_path, name, img, show)
        if preprocess:
            image = self.preprocess(image, stride)
        return image

    def preprocess(self, image, stride=None):
        if stride is None:
            stride = self.input_to_output_stride
        image = image[::stride, ::stride]
        """
        if self.insets[2] == 0:
            self.insets[2] = image.shape[0]
        if self.insets[3] == 0:
            self.insets[3] = image.shape[1]
        """
        print(self.insets)
        image = image[self.insets[1]:image.shape[0]-self.insets[3], self.insets[0]:image.shape[0]-self.insets[2]]
        print(image.shape)
        print(self.input_dim)
        image = image.reshape(tuple(np.concatenate(([1], np.array(self.input_dim)))))
        image = image.astype("float32")
        return image

    def create_erianet_base(self):
        input_d = self.input_dim
        print("Creating")
        print(input_d)
        seq = Sequential()
        #seq.add(Conv2D(filters=9, kernel_size=(3, 3), strides=(2, 2), activation='relu', input_shape=input_d))
        #seq.add(Flatten())
        seq.add(Dense(200, activation='linear', input_shape=input_d))
        seq.add(Dense(100, activation='linear'))
        seq.add(Dropout(0.2))
        seq.add(Dense(50, activation='linear'))

        """
        seq.add(Dense(600, input_shape=(input_d,), activation='linear'))
        seq.add(Dropout(0.2))
        seq.add(Dense(300, activation='linear'))
        seq.add(Dense(200, activation='linear'))
        seq.add(Dropout(0.1))
        seq.add(Dense(100, activation='linear'))
        seq.add(Dropout(0.2))
        seq.add(Dense(50, activation='linear'))
        """
        """
        for i in range(len(hidden_layer_size)):
            if i == 0:
                seq.add(Dense(hidden_layer_size[i], input_shape=(input_d,), activation='linear'))
            else:
                seq.add(Dense(hidden_layer_size[i], activation='linear'))
            seq.add(Dropout(0.2))"""
        return seq

    def create_erianet(self):
        # input_size = self.input_dim[0] * self.input_dim[1]
        input_d = self.input_dim
        print(input_d)
        print(tuple(input_d))
        input_a = Input(shape=tuple(input_d))
        input_b = Input(shape=tuple(input_d))
        base_network = self.create_erianet_base()
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        distance = Lambda(euclidean_distance, output_shape=euclidean_dist_output_shape)([processed_a, processed_b])
        model = Model(inputs=[input_a, input_b], outputs=distance)
        return model

    def gen_data_servantrain(self, train_set_size, class_folder_names, pic_dir, input_image_size=(100, 100),
                     input_to_output_stride=2):
        pass

    @deprecated
    def gen_data_new(self, train_set_size, class_folder_names, pic_dir, input_image_size=(100, 100),
                     input_to_output_stride=2):
        if input_image_size[0] % input_to_output_stride != 0 and input_image_size[1] % input_to_output_stride != 0:
            raise Exception("Input image size must be divisible by the stride")
        total_image_length = self.input_dim
        classes = len(class_folder_names)

        count = 0
        x_tr_positive = np.zeros(np.concatenate(([train_set_size, 2], total_image_length)))  # Save n pairs of images
        y_tr_positive = np.zeros([train_set_size, 1])  # Is this pair a positive or a negative

        # Gen Positive Examples
        for i in range(classes):  # From all classes
            for j in range(int(train_set_size / classes)):  # Get two different images of the same person
                ind1, ind2 = random_different_numbers(10)

                im1 = read_pgm(os.getcwd() + '/' + pic_dir + '/' + class_folder_names[i] + '/' + str(ind1 + 1) + '.pgm',
                               'rw+')
                im2 = read_pgm(os.getcwd() + '/' + pic_dir + '/' + class_folder_names[i] + '/' + str(ind2 + 1) + '.pgm',
                               'rw+')

                im1 = self.preprocess(im1)
                im2 = self.preprocess(im2)

                x_tr_positive[count, 0, :] = im1
                x_tr_positive[count, 1, :] = im2
                y_tr_positive[count] = 1
                count += 1

        count = 0
        x_tr_negative = np.zeros(np.concatenate(([train_set_size, 2], total_image_length)))
        y_tr_negative = np.zeros([train_set_size, 1])
        # Gen Negative Examples
        for i in range(int(train_set_size / 10)):  # Für ein Zehntel der Testdaten
            for j in range(10):  # Jeweils 10 Bilder auswählen
                ind1, ind2 = random_different_numbers(classes)

                im1 = read_pgm(os.getcwd() + '/' + pic_dir + '/' + class_folder_names[ind1] + '/' + str(j + 1) + '.pgm',
                               'rw+')
                im2 = read_pgm(os.getcwd() + '/' + pic_dir + '/' + class_folder_names[ind2] + '/' + str(j + 1) + '.pgm',
                               'rw+')

                im1 = self.preprocess(im1)
                im2 = self.preprocess(im2)

                x_tr_negative[count, 0, :] = im1
                x_tr_negative[count, 1, :] = im2
                y_tr_negative[count] = 0
                count += 1

        x_train = np.concatenate([x_tr_positive, x_tr_negative], axis=0) / 255  # Squish training-data from 0-255 to 0-1
        y_train = np.concatenate([y_tr_positive, y_tr_negative], axis=0)

        return x_train, y_train
