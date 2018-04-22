from robolib.networks.erianet import Erianet
from robolib.datamanager.siamese_data_loader import load_one_image
import os

epochs = 50 if os.path.exists("atnt.model") else 500
net = Erianet("atnt.model", input_image_size=(128, 128))
net.train("res_ModelData_AtnTFaces", 200)
net.save("atnt.model")

print("Train Finished!")

while True:
    name = input("Enter name:")
    img = int(input("Which image:"))

    if not os.path.exists("3BHIF/"+name):
        break

    image = load_one_image("3BHIF", name, img, True)
    probs = net.predict(image, "3BHIF")

    for pair in probs:
        print(pair[0], str(pair[1]), str(pair[2]))
