import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Input, Lambda, Dense, Dropout
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import re
import os
from os import path
from robolib.kerasplot.plot_callbacks import LossPlotCallback
from apps.facerecog.aifacerec_keras import contrastive_loss, read_pgm, gen_data_new


MODEL_FILENAME = "atnt4.model"
CLASS = 3
IMAGE = 3
REF_IMAGE = 7
NUM_CLASSES = 4


def load_image(cls, img):
    img = read_pgm("ModelData_AtnTFaces/s" + str(cls) + "/" + str(img) + ".pgm")
    img = img[::3, ::3]
    img = img.reshape(img.shape[0] * img.shape[1])
    img = img.astype("float32")
    return np.array([img])


def match_faces(input_img, ref_class, ref_image_index=4):
    reference_img = load_image(ref_class, ref_image_index)
    return float(model.predict([input_img, reference_img]))


def predict_face(input_img, ref_image_index=4):
    probabilities = np.array([], dtype=[('class', int), ('probability', float)])
    for i in range(0, NUM_CLASSES):
        probability = match_faces(input_img, i+1, REF_IMAGE)
        pair = (i+1, probability)
        probabilities = np.append(probabilities, np.array(pair, dtype=probabilities.dtype))
    probabilities = np.sort(probabilities, order='probability')
    return probabilities


model = load_model(MODEL_FILENAME, custom_objects={'contrastive_loss': contrastive_loss})

input_img = load_image(CLASS, IMAGE)

probs = predict_face(input_img)

print(probs)
print("Most probable face: "+str(probs[0][0])+" with a certainty of: "+str(probs[1][1] - probs[0][1]))
