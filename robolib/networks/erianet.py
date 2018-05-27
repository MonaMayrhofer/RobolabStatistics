# coding=utf-8
import os
import time
from os import path
from typing import Tuple, List

import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model

from robolib.images.pgmtools import read_pgm, RAW_IMAGE_EXTENSION
from robolib.networks.common import contrastive_loss, euclidean_dist_output_shape, euclidean_distance_tensor, \
    euclidean_distance_numeric
from robolib.networks.configurations import NetConfig
from robolib.networks.debug import debug_train_data
from robolib.networks.predict_result import PredictResult
from robolib.util.averager import Averager, ArithmeticAverager
from robolib.util.intermediate import load_intermediate, INTERMEDIATE_FILE_EXTENSION
from robolib.util.random import random_different_numbers


class Erianet:
    def __init__(self, model_path, config: type(NetConfig), input_image_size=(96, 128), input_to_output_stride=2,
                 insets=(0, 0, 0, 0), for_train=False):

        # = Sizes =
        self.input_image_size = input_image_size
        self.input_to_output_stride = input_to_output_stride
        self.insets = insets

        # = Config =
        self.config = config()
        self.base_network_input_dim = self.config.get_input_dim(input_image_size, input_to_output_stride, self.insets)

        # = Placeholders for Model =
        self.model = None
        self.base_network = None

        # = Status variables =
        self.is_blank = True
        self.for_train = for_train

        # = Initiate =
        self.create()
        if model_path is not None:
            assert path.isfile(model_path), 'Model file {0} could not be found.'.format(model_path)
            self.load(model_path)

    # =========TRAIN========

    def execute_train(self, train_data: Tuple[np.array, np.array], epochs: int, initial_epochs: int = None,
                      verbose: int = 1, callbacks=None) -> None:
        """
        Does the actual training.

        This function fits the model based on the data supplied.

        Parameters
        ----------
        train_data : Tuple[List, List]
            The training-data. The first entry should be a list containing the images, the second entry should be
            a list containing the labels.
        epochs : int
            For how many epochs shall the model be trained.
        initial_epochs : int, optional
            This is an override to 'epochs'. If this is specified, and the model is untrained (so no previous model
            was loaded) this value is used for 'epochs'.
        verbose: {0, 1, 2}
            0 = silent, 1 = progress bar, 2 = one line per epoch.
        callbacks: List
            An array of Keras callbacks, which is passed to model.fit

        """
        assert self.for_train, "If you intend to train this model, please specify it in the constructor 'for_train'"
        if initial_epochs is not None and self.is_blank:
            epochs = initial_epochs
        if callbacks is None:
            callbacks = []
        x_train, y_train = train_data
        self.is_blank = False
        self.model.fit([x_train[:, 0], x_train[:, 1]], y_train,
                       validation_split=0.1,
                       batch_size=128,
                       verbose=verbose,
                       epochs=epochs,
                       callbacks=callbacks)

    def train(self, data_folder: str, epochs: int = 100, initial_epochs: int = None, verbose: int = 2,
              train_set_size: int = 1000, callbacks: np.array = None):
        """
        Trains the model.

        Loads training data from 'data_folder' and trains the model on it.

        Parameters
        ----------
        data_folder : str
            The path to the training data.
        epochs : int
            The amount of runs this model shall train
        callbacks : np.array
            Forward these callbacks to model.train
        initial_epochs : int
            This is an override to 'epochs'. If this is specified, and the model is untrained (so no previous model
            was loaded) this value is used for 'epochs'.
        train_set_size : int
            How many different training-pairs shall be sampled
        verbose: {0, 1, 2}
            0 = silent, 1 = progress bar, 2 = one line per epoch.
        """
        train_data = self.get_train_data(train_set_size=train_set_size, data_folder=data_folder)
        self.execute_train(train_data=train_data, epochs=epochs,
                           callbacks=callbacks, initial_epochs=initial_epochs, verbose=verbose)

    def get_train_data(self, train_set_size, data_folder, class_folder_names=None, verbose=True) -> Tuple:
        """
        Generates training pairs.

        Generates a list of labelled positive and negative training pairs.

        Parameters
        ----------
        train_set_size : int
            How many Pairs shall be generated
        data_folder : str
            The path to the data-folder
        class_folder_names
            Use only these classes
        verbose : bool
            Print out status info?

        Returns
        -------
        A tuple of:
        1. A list of training-pairs
        2. A list of labels for the training pairs
        """
        if class_folder_names is None:
            class_folder_names = os.listdir(data_folder)

        classes = len(class_folder_names)
        examples_per_class = int(max(1.0, train_set_size / classes))

        total_image_length = self.base_network_input_dim
        x_shape = np.concatenate(([classes * examples_per_class, 2], total_image_length))
        y_shape = [classes * examples_per_class, 1]

        positive_x = np.zeros(x_shape)
        positive_y = np.zeros(y_shape)
        count = 0
        if verbose:
            print("Generating Positives")
        for i in range(classes):
            this_class_path = os.path.join(os.getcwd(), data_folder, class_folder_names[i])
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
        if verbose:
            print("Generating Negatives")
        for i in range(classes):
            first_class_path = os.path.join(os.getcwd(), data_folder, class_folder_names[i])
            first_class_images = os.listdir(first_class_path)
            used_classes = [i]
            for j in range(examples_per_class):
                other_ind = i
                while other_ind in used_classes and len(used_classes) < classes:
                    other_ind = np.random.randint(0, classes)
                used_classes.append(other_ind)

                other_class_path = os.path.join(os.getcwd(), data_folder, class_folder_names[other_ind])
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

        x_train = np.concatenate([positive_x, negative_x]) / 255  # Squish training-data from 0-255 to 0-1
        y_train = np.concatenate([positive_y, negative_y])

        return x_train, y_train

    def load_image_forwarded(self, file_path):
        extension = os.path.splitext(file_path)[1]
        if extension == INTERMEDIATE_FILE_EXTENSION:
            return load_intermediate(file_path)
        elif extension == RAW_IMAGE_EXTENSION:
            print("Unprocessed image found @ {0}".format(file_path))
            img = read_pgm(file_path)
            img = self.preprocess(img)
            return self.forward(img)
        else:
            raise Exception("Unknown file-extension '{0}' at '{1}'".format(extension, path))

    # ========= USE-Time Model Functions ========

    def forward(self, image):
        #print("Calling Tensorflow!")
        intermediate = self.base_network.predict(image)
        return intermediate

    def predict(self, input_img, reference_data_path, averager: type(Averager) = ArithmeticAverager):
        mon_start_time = time.time()

        # == Prepare input img ==
        input_img_forwarded = self.forward(self.preprocess(input_img))

        result = PredictResult()

        for person in os.listdir(reference_data_path):
            avg = averager()
            for image in os.listdir(os.path.join(reference_data_path, person)):
                reference_image_path = os.path.join(reference_data_path, person, image)
                reference_img_forwarded = self.load_image_forwarded(reference_image_path)
                distance = euclidean_distance_numeric((input_img_forwarded, reference_img_forwarded))
                avg += distance
            result.append(person, float(avg))
        distances = result.get()

        #print("Predict took: " + str(time.time() - mon_start_time))
        return distances

    # =========LOAD AND SAVE========

    def create(self):
        assert all(np.mod(self.input_image_size, self.input_to_output_stride) == (0, 0))
        self.base_network = self.config.create_base(self.base_network_input_dim)

        if self.for_train:
            input_a = Input(shape=tuple(self.base_network_input_dim))
            input_b = Input(shape=tuple(self.base_network_input_dim))
            processed_a = self.base_network(input_a)  # n-Dim classification Vector
            processed_b = self.base_network(input_b)  # n-Dim classification vector
            distance = Lambda(euclidean_distance_tensor, output_shape=euclidean_dist_output_shape)(
                [processed_a, processed_b])
            self.model = Model(inputs=[input_a, input_b], outputs=distance)
            optimizer = self.config.new_optimizer()
            self.model.compile(loss=contrastive_loss, optimizer=optimizer)

    def save(self, modelpath):
        print("Saving weights {0}".format(modelpath))
        self.base_network.save_weights(modelpath)

    def load(self, modelpath):
        assert self.base_network is not None, "Model must be created before loaded, if only weights are given."
        self.is_blank = False
        #print("Loading weights {0}".format(modelpath))
        self.base_network.load_weights(modelpath)

    # =========UTIL========

    def debug(self, data):
        debug_train_data(data, self.input_image_size, self.input_to_output_stride)

    def preprocess(self, image):

        assert self.input_image_size[0] * self.input_image_size[1] == image.shape[0] * image.shape[1], \
            "Images({0}) must have the same size as specified in input_image_size({1})".format(image.shape,
                                                                                               self.input_image_size)

        image = image[::self.input_to_output_stride, ::self.input_to_output_stride]
        image = image[self.insets[1]:image.shape[0] - self.insets[3], self.insets[0]:image.shape[1] - self.insets[2]]
        image = image.reshape(tuple(np.concatenate(([1], np.array(self.base_network_input_dim)))))
        image = image.astype("float32")
        return image
