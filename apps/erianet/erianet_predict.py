import matplotlib.pyplot as plt
import numpy as np
from robolib.images.pgmtools import read_pgm
from apps.erianet.erianet_util import load_erianet_model, get_3bhif_names

MODEL_FILENAME = "TestModel.model"
CLASS = 6
IMAGE = 3


def load_image(name, img, show=True, stride=2):
    img = read_pgm("3BHIF/" + name + "/" + str(img) + ".pgm")
    if show:
        plt.figure(1)
        plt.imshow(img, cmap='Greys_r')
        plt.show()
    img = img[::stride, ::stride]
    img = img.reshape(img.shape[0] * img.shape[1])
    img = img.astype("float32")
    return np.array([img])


def match_faces(input_img, ref_class, ref_image_index=4):
    reference_img = load_image(ref_class, ref_image_index, False)
    return float(model.predict([input_img, reference_img]))


def predict_face(input_img, ref_classes, ref_image_index=4):
    probabilities = np.array([], dtype=[('class', int), ('probability', float)])
    for i in range(0, len(ref_classes)):
        probability = match_faces(input_img, ref_classes[i], ref_image_index)
        pair = (i, probability)
        probabilities = np.append(probabilities, np.array(pair, dtype=probabilities.dtype))
    probabilities = np.sort(probabilities, order='probability')
    return probabilities


def predict_face_info(input_img, ref_classes, ref_image_index=4):
    probs = predict_face(input_img, ref_classes, ref_image_index)
    certainties = []
    biggestind = 0
    for i in range(len(probs)-1):
        certainty = probs[i+1][1]-probs[i][1]
        certainties.append([probs[i][0], probs[i][1], certainty])
        if certainties[biggestind][2] < certainty:
            biggestind = i
    return certainties[0:biggestind+1]


model = load_erianet_model(MODEL_FILENAME)


name = input("Enter name:")
img = int(input("Which image:"))

image = load_image(name, img)
names = get_3bhif_names()
probs = predict_face_info(image, names)

for pair in probs:
    print(names[pair[0]], str(pair[1]), str(pair[2]))
