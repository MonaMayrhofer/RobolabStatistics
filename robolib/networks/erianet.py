import os
from os import path
import numpy as np
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Lambda
from robolib.datamanager.siamese_data_loader import gen_data_new, load_one_image
from robolib.networks.common import contrastive_loss, euclidean_dist_output_shape, euclidean_distance
from keras import backend


class Erianet:
    def __init__(self, modelpath, input_image_size=(128, 128), input_to_output_stride=2, dont_init=False):
        self.input_image_size = input_image_size
        self.input_to_output_stride = input_to_output_stride
        self.model = None
        if not dont_init:
            if modelpath is None or not path.isfile(modelpath):
                self.create()
            else:
                self.load(modelpath)

    def train(self, data_folder, data_selection=None, callbacks=None, test_percent=0):
        if callbacks is None:
            callbacks = []
        if data_selection is None:
            data_selection = self.__get_names_of(data_folder)

        # TODO GEN_DATA_NEW TO HERE
        x, y = gen_data_new(1000, data_selection, data_folder, self.input_image_size, self.input_to_output_stride)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_percent)

        self.model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25, batch_size=128, verbose=2,
                       epochs=100,
                       callbacks=callbacks)

    def create(self, input_image_size=(128, 128), input_to_output_stride=2):
        self.model = self.__create_erianet(
            (input_image_size[0] / input_to_output_stride * input_image_size[1] / input_to_output_stride, 1))
        rms = RMSprop()
        self.model.compile(loss=contrastive_loss, optimizer=rms)

    def save(self, modelpath):
        print("Saving model to {}".format(modelpath))
        self.model.save(modelpath)

    def load(self, modelpath):
        print("Loading model from File {}".format(modelpath))
        self.model = load_model(modelpath, custom_objects={'contrastive_loss': contrastive_loss, 'backend': backend})

    def predict(self, input_img, reference_data_path, candidates=None):
        if candidates is None:
            candidates = self.__get_names_of(reference_data_path)
        probabilities = np.array([], dtype=[('class', int), ('probability', float)])
        for i in range(0, len(candidates)):
            reference_img = self.__load_image(reference_data_path, candidates[i], 1, False)  # TODO  REF IMAGE INDEX FOR
            probability = float(self.model.predict([input_img, reference_img]))
            pair = (i, probability)
            probabilities = np.append(probabilities, np.array(pair, dtype=probabilities.dtype))
        probabilities = np.sort(probabilities, order='probability')
        probs = probabilities
        certainties = []
        biggestind = 0
        for i in range(len(probs) - 1):
            certainty = probs[i + 1][1] - probs[i][1]
            certainties.append([candidates[probs[i][0]], probs[i][0], probs[i][1], certainty])
            if certainties[biggestind][2] < certainty:
                biggestind = i
        return certainties[0:biggestind + 1]

    @staticmethod
    def __get_names_of(folder):
        return next(os.walk(folder))[1]

    @staticmethod
    def __load_image(reference_path, name, img, show=False, stride=2):
        return load_one_image(reference_path, name, img, show, stride)

    @staticmethod
    def __create_erianet_base(input_d, hidden_layer_size):
        seq = Sequential()
        for i in range(len(hidden_layer_size)):
            if i == 0:
                seq.add(Dense(hidden_layer_size[i], input_shape=(input_d,), activation='linear'))
            else:
                seq.add(Dense(hidden_layer_size[i], activation='linear'))
            seq.add(Dropout(0.2))
        return seq

    @staticmethod
    def __create_erianet(input_dim):
        input_size = input_dim[0]*input_dim[1]
        hidden_layer_sizes = [200, 100, 50]
        input_a = Input(shape=(input_size,))
        input_b = Input(shape=(input_size,))
        base_network = Erianet.__create_erianet_base(input_size, hidden_layer_sizes)
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        distance = Lambda(euclidean_distance, output_shape=euclidean_dist_output_shape)([processed_a, processed_b])
        model = Model(inputs=[input_a, input_b], outputs=distance)
        return model