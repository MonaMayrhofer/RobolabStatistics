from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from robolib.robogui import pixel_editor
import cv2
import keras
import numpy as np
import os.path as path


DEBUG = False
RELEARN = False
MODEL_FILENAME = "KerasAdvancedModel"

# DATA STUFF
label_labels = ["Horizontal", "Vertikal"]
labels = np.random.randint(0, 2, size=(1000, 1))
size = 8
data = np.zeros(shape=(1000, size, size, 1))

for la, d in zip(labels, data):
    img = np.zeros((size, size))
    lineZ = np.random.randint(0, size)
    endLineZ = np.clip(lineZ + np.random.randint(-1, 2), 0, size)

    if la == 0:
        cv2.line(img, (0, lineZ), (size, endLineZ), 1.0)
    else:
        cv2.line(img, (lineZ, 0), (endLineZ, size), 1.0)

    # d[:] = np.reshape(img, (4*4, ))
    d[:, :, :] = np.reshape(img, (size, size, 1))

    if DEBUG:
        print(label_labels[la[0]])
        print(lineZ, endLineZ)
        print(img)


# MACHINE LEARNING STUFF

model = None
if RELEARN or not path.isfile(MODEL_FILENAME):
    print("Model will be recreated: File {} exists: {}".format(MODEL_FILENAME, path.isfile(MODEL_FILENAME)))
    model = Sequential()
    model.add(Conv2D(30, (3, 3), activation='relu', input_shape=(size, size, 1)))
    model.add(Conv2D(20, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(size*size, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
else:
    print("Loading model from File {}".format(MODEL_FILENAME))
    model = keras.models.load_model(MODEL_FILENAME)

one_hot_labels = keras.utils.to_categorical(labels, num_classes=2)
model.fit(data, one_hot_labels, epochs=300, batch_size=100)

print("Saving model to {}".format(MODEL_FILENAME))
model.save(MODEL_FILENAME)

while True:
    predict_data = [pixel_editor.get_pixel_input(size, size)]
    if all(1.0 not in row for row in predict_data):
        break

    if DEBUG:
        print(predict_data)
    output = model.predict(np.array(predict_data), 1, 3)
    if all(all(n < 0.9 for n in m) for m in output):
        print("Don't know, will guess: ")
    print(label_labels[np.argmax(output)])
    if DEBUG:
        print(np.around(output, 5))
