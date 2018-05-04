from robolib.networks.erianet import Erianet, ConvolutionalConfig, ClassicConfig, MutliConvConfig
from tensorflow.python.client import device_lib
import os

# ============= INPUT ============
print("Using devices: ")
print(device_lib.list_local_devices())

name = input("Enter the model's family-name: ")
start = input("Build on other model? [empty for None]: ")
if start == '':
    start = None
else:
    if not start.endswith(".model"):
        start += '.model'
    if not os.path.exists(start):
        print("File '{0}' does not exist.".format(start))
        exit(1)

runs = input("Enter runs: ")
runs = int(runs)

epochs_per_run = input("Enter epochs per run: ")
epochs_per_run = int(epochs_per_run)

train_folder = input("Enter image-folder: ")
if not os.path.exists(train_folder):
    print("Folder '{0}' couldn't be found!")
    exit(1)


# ============= TRAINING ============
net = Erianet(start, input_image_size=(96, 128), config=MutliConvConfig)
x_train, _, y_train, _ = net.prepare_train(train_folder)

for i in range(runs):
    print("==== RUN {0}/{1} ====".format(i, runs))
    net.execute_train(x_train, y_train, epochs_per_run)
    file_name = "{0}_{1}.model".format(name, (i+1)*epochs_per_run)
    print("==== SAVING {0} ====".format(file_name))
    net.save(file_name)
