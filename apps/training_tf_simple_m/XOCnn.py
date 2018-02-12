import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import robolib.robogui.pixel_editor as pe
import cv2
import robolib.images.feature_extraction as extr

DEBUG = True

label_labels = ["O", "X"]
labels = np.random.randint(0, 2, size=(1000, 1))
size = 9
data = np.zeros(shape=(1000, size, size, 1))

for la, d in zip(labels, data):
    img = np.empty((size, size))
    img.fill(-1)

    if la == 0:
        cv2.ellipse(img, (4, 4), (np.random.randint(2, 5), np.random.randint(2, 5)), 0, 360, 0, 1)
    else:
        randPointStart = np.random.randint(0, 16)
        randPointEnd = np.random.randint(0, 16)
        cv2.line(img, (int(randPointStart / 4), randPointStart % 4), (8 - int(randPointEnd / 4), 8 - randPointEnd % 4), 1)
        randPointStart = np.random.randint(0, 16)
        randPointEnd = np.random.randint(0, 16)
        cv2.line(img, (8 - int(randPointStart / 4), randPointStart % 4), (int(randPointEnd / 4), 8 - randPointEnd % 4), 1)

    img = extr.resize_image_to_info(img, size, size)

    d[:, :, :] = np.reshape(img, (size, size, 1))

    if DEBUG:
        if pe.show_image(img):
            DEBUG = False

model = Sequential()
model.add(Conv2D(9, (3, 3), activation='relu', input_shape=(size, size, 1)))
model.add(Conv2D(9, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), (2, 2)))
# model.add(Conv2D(3, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2), (2, 2)))
model.add(Flatten())
model.add(Dense(9, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

one_hot_labels = keras.utils.to_categorical(labels, num_classes=2)
model.fit(data, one_hot_labels, epochs=250, batch_size=80)

while True:
    predict_data = pe.get_drawing_input(size, size, size*3, size*3)

    if all(1.0 not in row for row in predict_data):
        break

    #    if DEBUG:
    pe.show_image(predict_data)

    output = model.predict(np.array([predict_data]), 1, 3)
    if all(all(n < 0.9 for n in m) for m in output):
        print("Don't know, will guess: ")
    print(label_labels[np.argmax(output)])
    if DEBUG:
        print(np.around(output, 5))
