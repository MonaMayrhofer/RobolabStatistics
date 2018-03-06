from os import path

from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from apps.wurschtnet.wurschtnet import create_wurschtnet, contrastive_loss, load_wurscht_model, get_3bhif_names
from robolib.datamanager.siamese_data_loader import gen_data_new

MODEL_FILENAME = "TestModel.model"
RELEARN = False

x, y = gen_data_new(1000, get_3bhif_names(), "3BHIF", input_image_size=(128, 128), input_to_output_stride=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
input_dim = x_train.shape[2]

if RELEARN or not path.isfile(MODEL_FILENAME):
    model = create_wurschtnet((input_dim, 1))
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
else:
    print("Loading model from File {}".format(MODEL_FILENAME))
    model = load_wurscht_model(MODEL_FILENAME)

model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25, batch_size=128, verbose=2, epochs=100,
          callbacks=[TensorBoard()])

print("Saving model to {}".format(MODEL_FILENAME))
model.save(MODEL_FILENAME)

pred_tr = model.predict([x_train[:, 0], x_train[:, 1]])
pred_ts = model.predict([x_test[:, 0], x_test[:, 1]])
