import numpy as np
from keras.models import Sequential
from keras.layers import Input, Lambda, Dense, Dropout
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from os import path
from robolib.kerasplot.plot_callbacks import LossPlotCallback
from apps.facerecog.aifacerec_keras import contrastive_loss, gen_data_new
import keras.backend as K

RELEARN = False
MODEL_FILENAME = "atnt4.model"
LOSS_PLOT = False
TENSORBOARD = True
EPOCHS = 1000


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def compute_accuracy(predictions, labels, thresh):
    return labels[predictions.ravel() < thresh].mean()


def create_base_network(input_d, hidden_layer_size):
    seq = Sequential()
    for i in range(len(hidden_layer_size)):
        if i == 0:
            seq.add(Dense(hidden_layer_size[i], input_shape=(input_d,), activation='linear'))
        else:
            seq.add(Dense(hidden_layer_size[i], activation='linear'))
        seq.add(Dropout(0.2))
    return seq


# get the data
samp_f = 3
total_to_samp = 10000
x, y = gen_data_new(samp_f, total_to_samp, classes=40)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.30)


# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
input_dim = x_train.shape[2]
input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))
hidden_layer_sizes = [200, 100, 50]
base_network = create_base_network(input_dim, hidden_layer_sizes)
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])


nb_epoch = EPOCHS
if RELEARN or not path.isfile(MODEL_FILENAME):
    model = Model(inputs=[input_a, input_b], outputs=distance)
    # train
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
else:
    print("Loading model from File {}".format(MODEL_FILENAME))
    model = load_model(MODEL_FILENAME, custom_objects={'contrastive_loss': contrastive_loss})

callbacks = []
if LOSS_PLOT:
    callbacks.append(LossPlotCallback())
if TENSORBOARD:
    callbacks.append(TensorBoard())
model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25,
          batch_size=128, verbose=2, epochs=nb_epoch, callbacks=callbacks)

print("Saving model to {}".format(MODEL_FILENAME))
model.save(MODEL_FILENAME)

# compute final accuracy on training and test sets
pred_tr = model.predict([x_train[:, 0], x_train[:, 1]])
pred_ts = model.predict([x_test[:, 0], x_test[:, 1]])
