import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Input, Lambda, Dense, Dropout
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import re
import os
from os import path
from robolib.kerasplot.plot_callbacks import LossPlotCallback

RELEARN = False
MODEL_FILENAME = "atnt4.model"


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


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


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
x, y = gen_data_new(samp_f, total_to_samp)
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


nb_epoch = 10
if RELEARN or not path.isfile(MODEL_FILENAME):
    model = Model(inputs=[input_a, input_b], outputs=distance)
    # train
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
else:
    print("Loading model from File {}".format(MODEL_FILENAME))
    model = load_model(MODEL_FILENAME, custom_objects={'contrastive_loss': contrastive_loss})


model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25,
          batch_size=128, verbose=2, epochs=nb_epoch, callbacks=[LossPlotCallback()])

print("Saving model to {}".format(MODEL_FILENAME))
model.save(MODEL_FILENAME)

# compute final accuracy on training and test sets
pred_tr = model.predict([x_train[:, 0], x_train[:, 1]])
pred_ts = model.predict([x_test[:, 0], x_test[:, 1]])

# auc and other things
tpr, fpr, _ = roc_curve(y_test, pred_ts)
roc_auc = auc(fpr, tpr)

plt.figure(1)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

thresh = .35
tr_acc = accuracy_score(y_train, (pred_tr < thresh).astype('float32'))
te_acc = accuracy_score(y_test, (pred_ts < thresh).astype('float32'))
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
print('* Mean of error less than  thresh (match): %0.3f' % np.mean(pred_ts[pred_ts < thresh]))
print('* Mean of error more than  thresh (no match): %0.3f' % np.mean(pred_ts[pred_ts >= thresh]))
print("* test case confusion matrix:")
print(confusion_matrix((pred_ts < thresh).astype('float32'), y_test))
plt.figure(2)
plt.plot(np.concatenate([pred_ts[y_test == 1], pred_ts[y_test == 0]]), 'bo')
plt.plot(np.ones(pred_ts.shape) * thresh, 'r')
plt.show()

