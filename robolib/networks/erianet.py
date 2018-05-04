import os
from os import path
import numpy as np
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Lambda, Conv2D, Flatten, BatchNormalization

from robolib.datamanager.siamese_data_loader import load_one_image
from robolib.networks.common import contrastive_loss, euclidean_dist_output_shape, euclidean_distance
from robolib.util.random import random_different_numbers
from keras import backend
from robolib.images.pgmtools import read_pgm
from robolib.util.decorations import deprecated
from robolib.networks.debug import debug_train_data, debug_image
import cv2
import time


class ClassicConfig:
    def __init__(self):
        pass

    def create_base(self, input_d):
        seq = Sequential()
        seq.add(Dense(200, activation='linear', input_shape=input_d))
        seq.add(Dense(100, activation='linear'))
        seq.add(Dropout(0.2))
        seq.add(Dense(50, activation='linear'))
        return seq

    def get_input_dim(self, input_image_size, input_to_output_stride, insets):
        return ((int(input_image_size[0] / input_to_output_stride) - insets[1] - insets[3]) *
                (int(input_image_size[1] / input_to_output_stride) - insets[0] - insets[2]),)


class ConvolutionalConfig:
    def __init__(self):
        pass

    def create_base(self, input_d):
        seq = Sequential()
        seq.add(Conv2D(filters=9, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=input_d))
        seq.add(Flatten())
        seq.add(Dense(200, activation='linear'))
        seq.add(Dense(100, activation='linear'))
        seq.add(Dropout(0.2))
        seq.add(Dense(50, activation='linear'))
        return seq

    def get_input_dim(self, input_image_size, input_to_output_stride, insets):
        return (int(input_image_size[0] / input_to_output_stride) - insets[1] - insets[3],
                int(input_image_size[1] / input_to_output_stride) - insets[0] - insets[2], 1)


class MutliConvConfig:
    def __init__(self):
        pass

    def create_base(self, input_d):
        seq = Sequential()
        seq.add(Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_d))
        seq.add(BatchNormalization())
        seq.add(Dropout(0.2))

        seq.add(Conv2D(8, (3, 3), activation='relu'))
        seq.add(BatchNormalization())
        seq.add(Dropout(0.2))

        seq.add(Conv2D(8, (3, 3), activation='relu'))
        seq.add(BatchNormalization())
        seq.add(Dropout(0.2))

        seq.add(Flatten())
        seq.add(Dense(500, activation='relu'))
        seq.add(Dropout(0.2))
        seq.add(Dense(500, activation='relu'))
        seq.add(Dense(50, activation='linear'))
        return seq

    def get_input_dim(self, input_image_size, input_to_output_stride, insets):
        return (int(input_image_size[0] / input_to_output_stride) - insets[1] - insets[3],
                int(input_image_size[1] / input_to_output_stride) - insets[0] - insets[2], 1)


class Erianet:
    def __init__(self, model_path, input_image_size=(128, 128), insets=(0, 0, 0, 0), input_to_output_stride=2,
                 do_not_init=False, config=None, experimental_preprocess=False):
        if config is None:
            config = ClassicConfig()
        else:
            config = config()
        self.config = config
        self.input_image_size = input_image_size
        self.input_to_output_stride = input_to_output_stride
        self.model = None
        self.insets = np.asarray(insets)
        self.input_dim = self.config.get_input_dim(input_image_size, input_to_output_stride, self.insets)
        self.model_path = model_path
        self.experimental_preprocess = experimental_preprocess
        if not do_not_init:
            if model_path is None or not path.isfile(model_path):
                self.create(input_image_size, input_to_output_stride)
            else:
                self.load(model_path)

    def prepare_train(self, data_folder, data_selection=None, servantrain=True, train_set_size=1000):
        x, y = self.get_train_data(train_set_size, data_folder, data_selection, servantrain=servantrain)
        return x, y

    def execute_train(self, x_train, y_train, epochs=100, callbacks=None, initial_epochs=None):
        if initial_epochs is not None and (self.model_path is None or not os.path.exists(self.model_path)):
            epochs = initial_epochs
        if callbacks is None:
            callbacks = []
        self.model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25, batch_size=128, verbose=2,
                       epochs=epochs,
                       callbacks=callbacks)

    def train(self, data_folder, epochs=100, data_selection=None, callbacks=None, initial_epochs=None,
              servantrain=True, train_set_size=1000):
        x_train, y_train = self.prepare_train(data_folder, data_selection, servantrain, train_set_size)
        self.execute_train(x_train, y_train, epochs, callbacks, initial_epochs)

    def get_train_data(self, amount, data_folder, data_selection=None, servantrain=True):
        if data_selection is None:
            data_selection = self.__get_names_of(data_folder)
        if servantrain:
            return self.gen_data_servantrain(amount, data_selection, data_folder)
        else:
            return self.gen_data_new(amount, data_selection, data_folder, self.input_image_size,
                                     self.input_to_output_stride)

    def create(self, input_image_size=(128, 128), input_to_output_stride=2):
        assert all(np.mod(input_image_size, input_to_output_stride) == (0, 0))
        self.model = self.create_erianet()

        rms = RMSprop()
        self.model.compile(loss=contrastive_loss, optimizer=rms)

    def save(self, modelpath):
        # print("Saving model to {}".format(modelpath))
        self.model.save(modelpath)

    def load(self, modelpath):
        # print("Loading model from File {}".format(modelpath))
        self.model = load_model(modelpath, custom_objects={'contrastive_loss': contrastive_loss, 'backend': backend})

    def compare(self, input_img, reference_path, reference_name, show=False, stride=None, preprocess=False):
        # Optimierungsideen:
        # Wenn Standardabweichung klein genug ist, den bis jetztigen Durchschnitt als gegeben annehmen
        reference_imgs = self.load_image(reference_path, reference_name, None, show=show, stride=stride,
                                         preprocess=preprocess)
        probability_sum = 0
        probability_amount = 0
        for reference_img in reference_imgs:
            probability_sum += float(self.model.predict([input_img, reference_img]))
            probability_amount += 1
        return probability_sum / probability_amount

    def predict(self, input_img, reference_data_path, candidates=None, give_all=False):
        mon_start_time = time.time()
        input_img = self.preprocess(input_img)

        if candidates is None:
            candidates = self.__get_names_of(reference_data_path)
        probabilities = np.array([], dtype=[('class', int), ('probability', float)])
        last = 0
        for i in range(0, len(candidates)):
            if time.time() - last > 1:
                last = time.time()
                print("{0:.1f}%".format(i/len(candidates)*100))

            probability = self.compare(input_img, reference_data_path, candidates[i], False, preprocess=True)
            pair = (i, probability)
            probabilities = np.append(probabilities, np.array(pair, dtype=probabilities.dtype))
        probabilities = np.sort(probabilities, order='probability')
        probs = probabilities
        certainties = []
        biggestind = 0
        for i in range(len(probs)):
            if i != len(probs) - 1:
                certainty = probs[i + 1][1] - probs[i][1]
            else:
                certainty = 0
            certainties.append([candidates[probs[i][0]], probs[i][0], probs[i][1], certainty])
            if certainties[biggestind][2] < certainty:
                biggestind = i
        print("Predict took: " + str(time.time() - mon_start_time))
        if give_all:
            return certainties
        return certainties[0:biggestind + 1]

    @staticmethod
    def __get_names_of(folder):
        assert os.path.isdir(folder), "Cannot find folder '{0}'".format(folder)
        return next(os.walk(folder))[1]

    def load_image(self, reference_path, name, img, show=False, stride=None, preprocess=False):
        if stride is None:
            stride = self.input_to_output_stride
        image = load_one_image(reference_path, name, img, show)
        if preprocess:
            if img is not None:
                image = self.preprocess(image, stride)
            else:
                image = [self.preprocess(currimg, stride) for currimg in image]
        return image

    def preprocess(self, image, stride=None):
        if stride is None:
            stride = self.input_to_output_stride

        assert (self.input_image_size[0] *
                self.input_image_size[1]) == \
               (image.shape[0] *
                image.shape[1]), \
            "Images({0}) must have the same size as specified in input_image_size({1})".format(image.shape,
                                                                                               self.input_image_size)

        image = image[::stride, ::stride]
        image = image[self.insets[1]:image.shape[0] - self.insets[3], self.insets[0]:image.shape[1] - self.insets[2]]
        # debug_image(image)
        if self.experimental_preprocess:
            image = cv2.Canny(image, 50, 150)
        # debug_image(image)
        image = image.reshape(tuple(np.concatenate(([1], np.array(self.input_dim)))))
        image = image.astype("float32")
        return image

    def create_erianet_base(self):
        ind = self.input_dim
        return self.config.create_base(ind)

    def create_erianet(self):
        input_a = Input(shape=tuple(self.input_dim))
        input_b = Input(shape=tuple(self.input_dim))
        base_network = self.create_erianet_base()
        processed_a = base_network(input_a)  # n-Dim classification Vector
        processed_b = base_network(input_b)  # n-Dim classification vector
        distance = Lambda(euclidean_distance, output_shape=euclidean_dist_output_shape)([processed_a, processed_b])
        model = Model(inputs=[input_a, input_b], outputs=distance)
        return model

    def debug(self, data):
        debug_train_data(data, self.input_image_size, self.input_to_output_stride)

    def gen_data_servantrain(self, train_set_size, class_folder_names, pic_dir, output=True):
        classes = len(class_folder_names)
        examples_per_class = int(max(1.0, train_set_size / classes))

        total_image_length = self.input_dim
        x_shape = np.concatenate(([classes * examples_per_class, 2], total_image_length))
        y_shape = [classes * examples_per_class, 1]

        positive_x = np.zeros(x_shape)
        positive_y = np.zeros(y_shape)
        count = 0
        if output:
            print("Generating Positives")
        for i in range(classes):
            this_class_path = os.path.join(os.getcwd(), pic_dir, class_folder_names[i])
            this_class_images = os.listdir(this_class_path)
            if len(this_class_images) < 2:
                continue
            for j in range(examples_per_class):
                i1, i2 = random_different_numbers(len(this_class_images))
                image_path1 = os.path.join(this_class_path, this_class_images[i1])
                image_path2 = os.path.join(this_class_path, this_class_images[i2])
                im1 = self.preprocess(read_pgm(image_path1))
                im2 = self.preprocess(read_pgm(image_path2))

                positive_x[count, 0, :] = im1
                positive_x[count, 1, :] = im2
                positive_y[count] = 1
                count += 1

        # Gen Negative Examples
        count = 0
        examples_per_class = int(max(1.0, train_set_size / classes))

        negative_x = np.zeros(x_shape)
        negative_y = np.zeros(y_shape)
        if output:
            print("Generating Negatives")
        for i in range(classes):
            first_class_path = os.path.join(os.getcwd(), pic_dir, class_folder_names[i])
            first_class_images = os.listdir(first_class_path)
            used_classes = [i]
            for j in range(examples_per_class):
                other_ind = i
                while other_ind in used_classes:
                    other_ind = np.random.randint(0, classes)
                used_classes.append(other_ind)

                other_class_path = os.path.join(os.getcwd(), pic_dir, class_folder_names[other_ind])
                other_class_images = os.listdir(other_class_path)

                i1 = np.random.randint(0, len(first_class_images))
                i2 = np.random.randint(0, len(other_class_images))

                image_path1 = os.path.join(first_class_path, first_class_images[i1])
                image_path2 = os.path.join(other_class_path, other_class_images[i2])

                im1 = self.preprocess(read_pgm(image_path1))
                im2 = self.preprocess(read_pgm(image_path2))

                negative_x[count, 0, :] = im1
                negative_x[count, 1, :] = im2
                negative_y[count] = 0
                count += 1

        x_train = np.concatenate([positive_x, negative_x], axis=0) / 255  # Squish training-data from 0-255 to 0-1
        y_train = np.concatenate([positive_y, negative_y], axis=0)

        return x_train, y_train

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

        print("Shape: ")
        print(x_tr_positive.shape)
        # Gen Positive Examples
        for i in range(classes):  # From all classes
            for j in range(int(train_set_size / classes)):  # Get two different images of the same person
                ind1, ind2 = random_different_numbers(10)

                im1 = read_pgm(os.getcwd() + '/' + pic_dir + '/' + class_folder_names[i] + '/' + str(ind1 + 1) + '.pgm')
                im2 = read_pgm(os.getcwd() + '/' + pic_dir + '/' + class_folder_names[i] + '/' + str(ind2 + 1) + '.pgm')

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

                im1 = read_pgm(os.getcwd() + '/' + pic_dir + '/' + class_folder_names[ind1] + '/' + str(j + 1) + '.pgm')
                im2 = read_pgm(os.getcwd() + '/' + pic_dir + '/' + class_folder_names[ind2] + '/' + str(j + 1) + '.pgm')

                im1 = self.preprocess(im1)
                im2 = self.preprocess(im2)

                x_tr_negative[count, 0, :] = im1
                x_tr_negative[count, 1, :] = im2
                y_tr_negative[count] = 0
                count += 1

        x_train = np.concatenate([x_tr_positive, x_tr_negative], axis=0) / 255  # Squish training-data from 0-255 to 0-1
        y_train = np.concatenate([y_tr_positive, y_tr_negative], axis=0)

        self.debug(x_train, )

        print("NewTrain-Shapes:")
        print(x_train.shape)
        print(y_train.shape)

        return x_train, y_train
