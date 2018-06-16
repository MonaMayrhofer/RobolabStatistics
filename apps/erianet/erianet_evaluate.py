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

colors = ['\033[92m', '\033[93m', '\033[93m',  '\033[91m', '\033[91m', '\033[91m', '\033[91m']

for person in os.listdir(input_set):
    print("=={0}==".format(person))
    for imgnr in os.listdir(os.path.join(input_set, person)):
        image = load_one_image(input_set, person, imgnr, False)
        names = net.predict(image, reference_set, debug=False)

        grade = 6
        for i in range(0, 5):
            if person == PredictResult.name(names[i]):
                grade = i
                break
        print(colors[grade]+" * {0}: {1}".format(imgnr, grade)+'\033[0m')
