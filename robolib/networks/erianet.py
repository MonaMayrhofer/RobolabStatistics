import os
import time
from os import path
from typing import Tuple, List

import numpy as np
from keras import backend
from keras.layers import Input, Lambda
from keras.models import Model, load_model

from robolib.datamanager.siamese_data_loader import load_one_image
from robolib.images.pgmtools import read_pgm
from robolib.networks.common import contrastive_loss, euclidean_dist_output_shape, euclidean_distance_tensor, \
    euclidean_distance_numeric
from robolib.networks.configurations import NetConfig
from robolib.networks.debug import debug_train_data
from robolib.util.random import random_different_numbers
from robolib.util.intermediate import load_intermediates


class Erianet:
    def __init__(self, model_path, config: type(NetConfig), input_image_size=(128, 128), input_to_output_stride=2,
                 do_not_init=False, load_only_weights=True, insets=(0, 0, 0, 0)):
        self.config = config()
        self.input_image_size = input_image_size
        self.input_to_output_stride = input_to_output_stride
        self.model = None
        self.base_network = None
        self.insets = insets
        self.input_dim = self.config.get_input_dim(input_image_size, input_to_output_stride, self.insets)
        self.model_path = model_path
        if not do_not_init:
            if model_path is None or not path.isfile(model_path):
                self.create(input_image_size, input_to_output_stride)
                self.is_blank = True
            else:
                if load_only_weights:
                    self.create(input_image_size, input_to_output_stride)
                self.load(model_path)
                self.is_blank = False

    # =========TRAIN========

    def execute_train(self, train_data: Tuple[np.array, np.array], epochs: int,
                      initial_epochs: int = None, verbose: int = 1,
                      callbacks=None, batch_size=128) -> None:
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
        batch_size:
            Passthrough parameter to model.fit

        """
        if initial_epochs is not None and self.is_blank:
            epochs = initial_epochs
        if callbacks is None:
            callbacks = []
        x_train, y_train = train_data
        print(type(train_data))
        print("Calling Tensorflow")
        self.model.fit([x_train[:, 0], x_train[:, 1]], y_train,
                       validation_split=0.1,
                       batch_size=batch_size,
                       verbose=verbose,
                       epochs=epochs,
                       callbacks=callbacks)

    def train(self, data_folder: str, epochs: int = 100, initial_epochs: int = None, verbose: int = 2,
              train_set_size: int = 1000, callbacks: np.array = None, batch_size: int = 128):
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
        batch_size : int
            Parameter is forwarded to model.train
        verbose : {0, 1, 2}
        """
        train_data = self.get_train_data(train_set_size=train_set_size, data_folder=data_folder)
        self.execute_train(train_data=train_data, epochs=epochs,
                           callbacks=callbacks, initial_epochs=initial_epochs,
                           batch_size=batch_size, verbose=verbose)

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

        total_image_length = self.input_dim
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

        x_train = np.concatenate([positive_x, negative_x], axis=0) / 255  # Squish training-data from 0-255 to 0-1
        y_train = np.concatenate([positive_y, negative_y], axis=0)

        return x_train, y_train

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
        image = image.reshape(tuple(np.concatenate(([1], np.array(self.input_dim)))))
        image = image.astype("float32")
        return image

    # =========LOAD AND SAVE========

    def create(self, input_image_size=(128, 128), input_to_output_stride=2):
        assert all(np.mod(input_image_size, input_to_output_stride) == (0, 0))
        self.model, self.base_network = self.create_erianet()

        optimizer = self.config.new_optimizer()
        self.model.compile(loss=contrastive_loss, optimizer=optimizer)

    def create_erianet(self):
        input_a = Input(shape=tuple(self.input_dim))
        input_b = Input(shape=tuple(self.input_dim))
        base_network = self.config.create_base(self.input_dim)
        processed_a = base_network(input_a)  # n-Dim classification Vector
        processed_b = base_network(input_b)  # n-Dim classification vector
        distance = Lambda(euclidean_distance_tensor, output_shape=euclidean_dist_output_shape)(
            [processed_a, processed_b])
        model = Model(inputs=[input_a, input_b], outputs=distance)
        return model, base_network

    def save(self, modelpath, weights_only=True):
        if weights_only:
            print("Saving weights {0}".format(modelpath))
            self.model.save_weights(modelpath)
        else:
            self.model.save(modelpath)

    def load(self, modelpath, weights_only=True):
        # print("Loading model from File {}".format(modelpath))
        if weights_only:
            assert self.model is not None, "Model must be created before loaded, if only weights are given."
            print("Loading weights {0}".format(modelpath))
            self.model.load_weights(modelpath)
        else:
            self.model = load_model(modelpath,
                                    custom_objects={'contrastive_loss': contrastive_loss, 'backend': backend})

    # =========PREDICT========

    def forward(self, image):
        print("Calling Tensorflow!")
        intermediate = self.base_network.predict(image)
        return intermediate

    def distance(self, image_a, image_b, a_is_intermediate=False, b_is_intermediate=False):
        if not a_is_intermediate:
            image_a = self.forward(image_a)
        if not b_is_intermediate:
            image_b = self.forward(image_b)

        assert image_a.shape == image_b.shape
        distance = euclidean_distance_numeric((image_a, image_b))
        return distance

    def compare(self, input_img, reference_path, reference_name, input_img_intermediate=False,
                reference_img_intermediate=False, show=False, stride=None, preprocess=False):
        # Optimierungsideen:
        # Wenn Standardabweichung klein genug ist, den bis jetztigen Durchschnitt als gegeben annehmen
        if reference_img_intermediate:
            reference_imgs = load_intermediates(reference_path, reference_name)
        else:
            reference_imgs = self.load_image(reference_path, reference_name, None, show=show, stride=stride,
                                             preprocess=preprocess)
        probability_sum = 0
        probability_amount = 0
        for reference_img in reference_imgs:
            probability_sum += float(self.distance(image_a=input_img, image_b=reference_img,
                                                   a_is_intermediate=input_img_intermediate,
                                                   b_is_intermediate=reference_img_intermediate))
            probability_amount += 1
        return probability_sum / probability_amount

    def predict(self, input_img, reference_data_path, candidates=None, give_all=False, verbose=False,
                input_img_intermediate=False, reference_img_intermediate=False,):
        mon_start_time = time.time()
        input_img = self.preprocess(input_img)

        if candidates is None:
            candidates = os.listdir(reference_data_path)
        probabilities = np.array([], dtype=[('class', int), ('probability', float)])
        last = 0

        if not input_img_intermediate:
            input_img = self.forward(input_img)
        for i in range(0, len(candidates)):
            if time.time() - last > 1:
                last = time.time()
                print("{0:.1f}%".format(i / len(candidates) * 100))

            probability = self.compare(input_img, reference_data_path, candidates[i],
                                       input_img_intermediate=True,
                                       reference_img_intermediate=reference_img_intermediate,
                                       show=False, preprocess=True)
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
        if verbose:
            print("Predict took: " + str(time.time() - mon_start_time))
        if give_all:
            return certainties
        return certainties[0:biggestind + 1]

    # =========UTIL========

    def debug(self, data):
        debug_train_data(data, self.input_image_size, self.input_to_output_stride)
