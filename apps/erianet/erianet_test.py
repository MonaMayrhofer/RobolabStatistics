from robolib.networks.erianet import Erianet
from robolib.networks.configurations import VGG19simplified
from robolib.datamanager.siamese_data_loader import load_one_image
from robolib.networks.predict_result import PredictResult
import robolib.datamanager.datadir as datadir
import os

intermediate = True

input_set = datadir.get_converted_dir("3BHIF")
reference_set = datadir.get_intermediate_dir("i3BHIFsimplified")

model_name = "vggsimple_5000_1528508930050.model"

net = Erianet(datadir.get_model_dir(model_name), input_image_size=(96, 128), config=VGG19simplified)

print("Train Finished!")

while True:
    name = input("Enter name of {0}: ".format(input_set))
    if name == '':
        break
    img = input("Which image: ")
    if img == '':
        break
    img = int(img)

    if not os.path.exists(os.path.join(input_set, name)):
        break

    image = load_one_image(input_set, name, img, True)
    names = net.predict(image, reference_set, debug=True)

    for name in names:
        print(PredictResult.name(name), PredictResult.difference(name))
