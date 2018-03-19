from robolib.networks.erianet import Erianet
from robolib.datamanager.siamese_data_loader import load_one_image

net = Erianet("TestModel.model", input_image_size=(92, 112))
net.train("ModelData_AtnTFaces")

print("Train Finished!")
name = input("Enter name:")
img = int(input("Which image:"))

image = load_one_image("ModelData_AtnTFaces", name, img, True)
probs = net.predict(image, "ModelData_AtnTFaces")

for pair in probs:
    print(pair[0], str(pair[1]), str(pair[2]))

