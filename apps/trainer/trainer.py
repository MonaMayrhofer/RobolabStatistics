from robolib.networks.erianet import Erianet
from tensorflow.python.client import device_lib
import os
import argparse
import importlib
from robolib.util.model_uuid import get_model_identifier
import robolib.datamanager.datadir as datadir

__ROBOLIB_CONFIG_PACKAGE = "robolib.networks.configurations"
MODEL_EXTENSION = ".model"


def main():
    # ============= ARGUMENTS ===========
    parser = argparse.ArgumentParser(description='Train Erianet')
    parser.add_argument('--name', '-n', type=str, nargs='?', help='The prefix for the trained model file?')
    parser.add_argument('--base', '-b', type=str, nargs='?',
                        help='Shall this model be trained ontop of existing one? [Optional]')
    parser.add_argument('--runs', '-r', type=int, nargs='?', help='How many runs (files) shall be made?')
    parser.add_argument('--epochs', '-e', type=int, nargs='?', help='Each run shall last for how many epochs?')
    parser.add_argument('--folder', '-f', type=str, nargs='?', help='Where are the training-images?')
    parser.add_argument('--config', '-c', type=str, nargs='?', help='Which configuration shall be used?')

    args = parser.parse_args()
    name = args.name
    start = args.base
    runs = args.runs
    epochs_per_run = args.epochs
    train_folder = args.folder
    config_name = args.config

    if None in [name, runs, epochs_per_run, train_folder, config_name]:
        print("Please specify all arguments (--help)")
        exit(1)

    config = getattr(importlib.import_module(__ROBOLIB_CONFIG_PACKAGE), config_name)

    train(start, train_folder, runs, epochs_per_run, name, config)


# ============= TRAINING ============
def train(start, train_folder, runs, epochs_per_run, name, config):
    start = datadir.get_model_dir(start) if start is not None else None
    train_folder = datadir.get_converted_dir(train_folder)

    print("Available devices: ")
    for dev in device_lib.list_local_devices():
        print("{0:5} {1:20} {2}".format(dev.device_type, dev.name, dev.physical_device_desc))

    print("== Starting Training ==")
    net = Erianet(start, config, input_image_size=(96, 128), for_train=True)
    x_train, y_train = net.get_train_data(train_set_size=2000, data_folder=train_folder)

    for i in range(runs):
        print("==== RUN {0}/{1} ====".format(i, runs))
        net.execute_train((x_train, y_train), epochs_per_run)
        uuid = get_model_identifier((i+1)*epochs_per_run)
        file_name = "{0}_{1}_{2}.model".format(name, (i+1)*epochs_per_run, uuid)
        file_name = datadir.get_model_dir(file_name)
        print("==== SAVING {0} ====".format(file_name))
        net.save(file_name)


if __name__ == "__main__":
    main()
