import numpy as np
import os

INTERMEDIATE_FILE_EXTENSION = ".interm"

def save_intermediate(intermediate, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    file = open(path, 'wb')
    np.save(file, intermediate)


def load_intermediate(path):
    file = open(path, 'rb')
    arr = np.load(file)
    return arr


def load_intermediates(reference_path, name):
    return [load_intermediate(os.path.join(reference_path, name, image))
            for image in os.listdir(os.path.join(reference_path, name))]
