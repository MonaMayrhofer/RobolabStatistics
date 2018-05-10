from robolib.networks.erianet import Erianet, ConvolutionalConfig, ClassicConfig, MultiConvConfig, VGG19ish
import robolib.networks.erianet
from tensorflow.python.client import device_lib
import os
import argparse
import importlib
__ROBOLIB_CONFIG_PACKAGE = "robolib.networks.erianet"
MODEL_EXTENSION = ".hd5"

def main():
    # ============= ARGUMENTS ===========
    parser = argparse.ArgumentParser(description='Train Erianet')
    parser.add_argument('--name', '-n', type=str, nargs='?', help='The prefix for the trained model file?')
    parser.add_argument('--base', '-b', type=str, nargs='?', help='Shall this model be trained ontop of existing one?')
    parser.add_argument('--runs', '-r', type=str, nargs='?', help='How many runs (files) shall be made?')
    parser.add_argument('--epochs', '-e', type=str, nargs='?', help='Each run shall last for how many epochs?')
    parser.add_argument('--folder', '-f', type=str, nargs='?', help='Where are the training-images?')
    parser.add_argument('--config', '-c', type=str, nargs='?', help='Which configuration shall be used?')

    args = parser.parse_args()
    name = args.name
    start = args.base
    runs = args.runs
    epochs_per_run = args.epochs
    train_folder = args.folder
    config_name = args.config
    # ============= INPUT ============

    if name is None:
        name = input("Enter the model's family-name [or specify -n]: ")

    if start == '':
        start = None
    elif start is not None:
        if not start.endswith(MODEL_EXTENSION):
            start += MODEL_EXTENSION
        if not os.path.exists(start):
            print("File '{0}' does not exist.".format(start))
            exit(1)

    if runs is None:
        runs = input("Enter runs [or specify -r]: ")
    runs = int(runs)

    if epochs_per_run is None:
        epochs_per_run = input("Enter epochs per run [or specify -e]: ")
    epochs_per_run = int(epochs_per_run)

    if train_folder is None:
        train_folder = input("Enter image-folder [or specify -f]: ")
    if not os.path.exists(train_folder):
        print("Folder '{0}' couldn't be found!".format(train_folder))
        exit(1)

    config = getattr(importlib.import_module(__ROBOLIB_CONFIG_PACKAGE), config_name)

    train(start, train_folder, runs, epochs_per_run, name, config)


# ============= TRAINING ============
def train(start, train_folder, runs, epochs_per_run, name, config, model_dir="models"):
    print("Available devices: ")
    for dev in device_lib.list_local_devices():
        print("{0:5} {1:20} {2}".format(dev.device_type, dev.name, dev.physical_device_desc))

    print("== Starting Training ==")
    net = Erianet(start, input_image_size=(96, 128), config=config)
    x_train, y_train = net.prepare_train(train_folder, train_set_size=4000)

    for i in range(runs):
        print("==== RUN {0}/{1} ====".format(i, runs))
        net.execute_train(x_train, y_train, epochs_per_run, validation_split=0.128)
        file_name = "{0}_{1}.model".format(name, (i+1)*epochs_per_run)
        file_name = os.path.join(model_dir, file_name)
        print("==== SAVING {0} ====".format(file_name))
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        net.save(file_name)


if __name__ == "__main__":
    main()
