from robolib.networks.erianet import Erianet, ConvolutionalConfig, ClassicConfig
from robolib.datamanager.siamese_data_loader import load_one_image
import os

servantrain = True

train_set = "convlfw" if servantrain else "96128res_3BHIF"
predict_set = "res96128_ModelData_AtnTFaces"
model_name = "atnt_2500.model"

net = Erianet(model_name, input_image_size=(96, 128), config=ConvolutionalConfig)
# net.train(train_set, 10, initial_epochs=200, servantrain=servantrain)
# net.save(model_name)

print("Train Finished!")

while True:
    name = input("Enter name of {0}:".format(predict_set))
    if name == '':
        break
    img = input("Which image:")
    if img == '':
        break
    img = int(img)

    if not os.path.exists(os.path.join(predict_set, name)):
        break

    image = load_one_image(predict_set, name, img, True)
    probs = net.predict(image, predict_set)

    for pair in probs:
        print(pair[0], str(pair[1]), str(pair[2]))
