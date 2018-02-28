import numpy as np
from keras.models import load_model

import robolib.datamanager.atntfaces as data
from apps.facerecog.aifacerec_keras import contrastive_loss, read_pgm
import matplotlib.pyplot as plt

data.get_data("AtnTFaces", True)

MODEL_FILENAME = "atnt4.model"
CLASS = 6
IMAGE = 3


def load_image(cls, img):
    img = read_pgm("ModelData_AtnTFaces/s" + str(cls) + "/" + str(img) + ".pgm")
    img = img[::3, ::3]
    img = img.reshape(img.shape[0] * img.shape[1])
    img = img.astype("float32")
    return np.array([img])


def match_faces(input_img, ref_class, ref_image_index=4):
    reference_img = load_image(ref_class, ref_image_index)
    return float(model.predict([input_img, reference_img]))


def predict_face(input_img, num_classes, ref_image_index=4):
    probabilities = np.array([], dtype=[('class', int), ('probability', float)])
    for i in range(0, num_classes):
        probability = match_faces(input_img, i+1, ref_image_index)
        pair = (i+1, probability)
        probabilities = np.append(probabilities, np.array(pair, dtype=probabilities.dtype))
    probabilities = np.sort(probabilities, order='probability')
    return probabilities


model = load_model(MODEL_FILENAME, custom_objects={'contrastive_loss': contrastive_loss})


for i in range(1, 41):
    input_img = load_image(i, IMAGE)

    probs = predict_face(input_img, 40)
    certaintyA = probs[1][1] - probs[0][1]
    certaintyB = probs[2][1] - probs[1][1]

    print("Predicted for: "+str(i)+"Most probable face: "+str(probs[0][0])+" with a certainty of: "+str(certaintyA)+" - "+str(certaintyB)+" = "+str(certaintyA - certaintyB))
    if i != probs[0][0]:
        print("===== WRONG =====")
        print(probs)
