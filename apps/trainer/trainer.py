from robolib.networks.erianet import Erianet, ConvolutionalConfig, ClassicConfig, MultiConvConfig, VGG19ish
from tensorflow.python.client import device_lib
import os
import argparse


def main():
    # ============= ARGUMENTS ===========
    parser = argparse.ArgumentParser(description='Train Erianet')
    parser.add_argument('--name', '-n', type=str, nargs='?', help='The prefix for the trained model file?')
    parser.add_argument('--base', '-b', type=str, nargs='?', help='Shall this model be trained ontop of existing one?')
    parser.add_argument('--runs', '-r', type=str, nargs='?', help='How many runs (files) shall be made?')
    parser.add_argument('--epochs', '-e', type=str, nargs='?', help='Each run shall last for how many epochs?')
    parser.add_argument('--folder', '-f', type=str, nargs='?', help='Where are the training-images?')

    args = parser.parse_args()
    name = args.name
    start = args.base
    runs = args.runs
    epochs_per_run = args.epochs
    train_folder = args.folder
    # ============= INPUT ============

    print("Available devices: ")
    for dev in device_lib.list_local_devices():
        print("{0:5} {1:20} {2}".format(dev.device_type, dev.name, dev.physical_device_desc))

    if name is None:
        name = input("Enter the model's family-name [or specify -n]: ")

    if start == '':
        start = None
    elif start is not None:
        if not start.endswith(".model"):
            start += '.model'
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

    train(start, train_folder, runs, epochs_per_run, name)


# ============= TRAINING ============
def train(start, train_folder, runs, epochs_per_run, name):
    print("== Starting Training ==")
    net = Erianet(start, input_image_size=(96, 128), config=VGG19ish)
    x_train, y_train = net.prepare_train(train_folder, train_set_size=4000)

    for i in range(runs):
        print("==== RUN {0}/{1} ====".format(i, runs))
        net.execute_train(x_train, y_train, epochs_per_run, validation_split=0.128, batch_size=256)
        file_name = "{0}_{1}.model".format(name, (i+1)*epochs_per_run)
        print("==== SAVING {0} ====".format(file_name))
        net.save(file_name)


if __name__ == "__main__":
    main()
