import os

import numpy as np
import matplotlib.pyplot as plt
from robolib.images.pgmtools import read_pgm
from robolib.util.random import random_different_numbers
from robolib.networks.debug import debug_image
import random
import warnings


def load_one_image(referenceimgpath, name, img, show=False, amount=5):
    if img is not None:
        path = os.path.join(referenceimgpath, name, str(img))
        if not path.endswith(".pgm"):
            path += ".pgm"
        img = read_pgm(path)
        if show:
            debug_image(img)
        return img
    imgs = []
    selection = os.listdir(os.path.join(referenceimgpath, name))
    selection = random.sample(selection, min(len(selection), amount))
    for file in selection:
        img = read_pgm(os.path.join(referenceimgpath, name, file))
        imgs.append(img)
    if show:
        warnings.warn("Show is not yet supported with bulk-load", RuntimeWarning)  # TODO Show with bulk-load
    return imgs

