from keras.models import Sequential
from keras.layers import Dense
from robolib.robogui import pixel_editor
import keras
import numpy as np

# DATA STUFF
label_labels = ["Horizontal", "Vertikal"]
labels = np.random.randint(0, 2, size=(1000, 1))
data = np.zeros(shape=(1000, 4))

for la, d in zip(labels, data):
    if la == 0:
        if np.random.randint(0, 2) == 0:
            d[0] = 1.0
            d[1] = 1.0
        else:
            d[2] = 1.0
            d[3] = 1.0
    else:
        if np.random.randint(0, 2) == 0:
            d[0] = 1.0
            d[2] = 1.0
        else:
            d[1] = 1.0
            d[3] = 1.0

#MACHINE LEARNING STUFF

model = Sequential()
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
    predict_data = np.reshape(pixel_editor.get_pixel_input(2, 2), (4, ))
    if 1.0 not in predict_data:
        break
    print(predict_data)
    output = model.predict(np.array([predict_data]), 1, 3)
    print(label_labels[np.argmax(output)])
    print(np.around(output, 1))
