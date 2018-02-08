from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np

# DATA STUFF
labels = np.random.randint(0, 1, size=(1000, 1))
data = np.zeros(shape=(1000, 4))

for la, d in zip(labels, data):
    if la == 0:
        d[[0, 1]] = [1.0, 1.0]
    else:
        d[[0, 2]] = [1.0, 1.0]

predict_data = [1.0, 1.0, 0.0, 0.0]

#MACHINE LEARNING STUFF

model = Sequential()
model.add(Dense(4, activation='relu', input_dim=4))
model.add(Dense(2, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

one_hot_labels = keras.utils.to_categorical(labels, num_classes=2)
model.fit(data, one_hot_labels, epochs=100, batch_size=32)

output = model.predict(np.array([predict_data]), 1, 3)
print(output)
