from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from robolib.robogui import pixel_editor
import cv2
import keras
import numpy as np


DEBUG = False

# DATA STUFF
label_labels = ["Horizontal", "Vertikal"]
labels = np.random.randint(0, 2, size=(1000, 1))
data = np.zeros(shape=(1000, 4, 4, 1))

for la, d in zip(labels, data):
    img = np.zeros((4, 4))
    lineZ = np.random.randint(0, 4)
    endLineZ = np.clip(lineZ + np.random.randint(-1, 2), 0, 4)

    if la == 0:
        cv2.line(img, (0, lineZ), (4, endLineZ), 1.0)
    else:
        cv2.line(img, (lineZ, 0), (endLineZ, 4), 1.0)

    # d[:] = np.reshape(img, (4*4, ))
    d[:, :, :] = np.reshape(img, (4, 4, 1))

    if DEBUG:
        print(label_labels[la[0]])
        print(lineZ, endLineZ)
        print(img)


# MACHINE LEARNING STUFF

model = Sequential()
model.add(Conv2D(4, (2, 2), activation='relu', input_shape=(4, 4, 1)))
model.add(Conv2D(4, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(4, activation='relu', input_dim=4))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

one_hot_labels = keras.utils.to_categorical(labels, num_classes=2)
model.fit(data, one_hot_labels, epochs=400, batch_size=50)

while True:
    predict_data = [pixel_editor.get_pixel_input(4, 4)]
    if all(1.0 not in row for row in predict_data):
        break

    if DEBUG:
        print(predict_data)
    output = model.predict(np.array(predict_data), 1, 3)
    print(label_labels[np.argmax(output)])
    if DEBUG:
        print(np.around(output, 1))
