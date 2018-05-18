from robolib.networks.erianet import Erianet
from robolib.networks.configurations import VGG19ish
from robolib.datamanager.siamese_data_loader import load_one_image
import os

intermediate = True

input_set = "conv3BHIF"
reference_set = "interm3BHIF" if intermediate else input_set

model_name = "test_100_1526669017001.model"

net = Erianet(model_name, input_image_size=(96, 128), config=VGG19ish)

print("Train Finished!")

while True:
    name = input("Enter name of {0}:".format(input_set))
    if name == '':
        break
    img = input("Which image:")
    if img == '':
        break
    img = int(img)

    if not os.path.exists(os.path.join(input_set, name)):
        break

    image = load_one_image(input_set, name, img, True)
    probs = net.predict(image, reference_set, input_img_intermediate=False, reference_img_intermediate=intermediate,
                        verbose=True)

    for pair in probs:
        print(pair[0], str(pair[1]), str(pair[2]))
