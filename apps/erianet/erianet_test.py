from robolib.networks.erianet import Erianet
from robolib.datamanager.siamese_data_loader import load_one_image

net = Erianet("TestModel.model")
net.train("3BHIF")

print("Train Finished!")
name = input("Enter name:")
img = int(input("Which image:"))

image = load_one_image("3BHIF", name, img, True)
probs = net.predict(image, "3BHIF")

for pair in probs:
    print(pair[0], str(pair[1]), str(pair[2]))

