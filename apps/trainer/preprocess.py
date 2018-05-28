from robolib.networks.erianet import Erianet
from tensorflow.python.client import device_lib
from robolib.datamanager.siamese_data_loader import load_one_image
import os
import argparse
import importlib
from robolib.util.intermediate import save_intermediate, load_intermediate
import robolib.datamanager.datadir as datadir
import numpy as np

__ROBOLIB_CONFIG_PACKAGE = "robolib.networks.configurations"
INTERMEDIATE_EXTENSION = "interm"


def main():
    # ============= ARGUMENTS ===========
    parser = argparse.ArgumentParser(description='Train Erianet')
    parser.add_argument('--model', '-m', type=str, nargs='?', help='The path to the model?')
    parser.add_argument('--folder', '-f', type=str, nargs='?', help='The folder to be preprocessed')
    parser.add_argument('--config', '-c', type=str, nargs='?', help='Which configuration shall be used?')
    parser.add_argument('--output', '-o', type=str, nargs='?', help='Output directory name')

    args = parser.parse_args()
    model = args.model
    folder = args.folder
    config = args.config
    output = args.output

    config = getattr(importlib.import_module(__ROBOLIB_CONFIG_PACKAGE), config)

    process(model, folder, config, output)


# ============= TRAINING ============
def process(model, folder, config, output):
    print("Available devices: ")
    for dev in device_lib.list_local_devices():
        print("{0:5} {1:20} {2}".format(dev.device_type, dev.name, dev.physical_device_desc))

    print("== Starting Preprocess ==")
    net = Erianet(datadir.get_model_dir(model), config, input_image_size=(96, 128))

    for person in os.listdir(folder):
        print(" - {0}".format(person))
        for image_name in os.listdir(os.path.join(folder, person)):
            image = load_one_image(folder, person, image_name, False)
            intermediate = net.forward(net.preprocess(image))
            intermediate_path = os.path.join(output, person, "{0}.{1}".format(image_name, INTERMEDIATE_EXTENSION))
            save_intermediate(intermediate, intermediate_path)


if __name__ == "__main__":
    main()
