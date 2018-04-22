from robolib.networks.erianet import Erianet, ConvolutionalConfig
from robolib.datamanager.siamese_data_loader import load_one_image
import os

train_set = "res_ModelData_AtnTFaces"
predict_set = "res_ModelData_AtnTFaces"
model_name = "atnt.model"

net = Erianet(model_name, input_image_size=(128, 128), config=ConvolutionalConfig)
net.train(train_set, 10, initial_epochs=200)
net.save(model_name)

print("Train Finished!")

while True:
    name = input("Enter name:")
    img = int(input("Which image:"))

    if not os.path.exists(os.path.join(predict_set,name)):
        break

    image = load_one_image(predict_set, name, img, True)
    probs = net.predict(image, predict_set)

    for pair in probs:
        print(pair[0], str(pair[1]), str(pair[2]))
