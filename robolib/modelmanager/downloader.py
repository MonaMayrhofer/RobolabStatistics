import os.path
from urllib.request import urlretrieve


def get_model(url, filename):
    if os.path.isfile(filename):
        print("{} already present, to re-download it remove the file.".format(filename))
    else:
        print("{} not found, downloading it from: {} ".format(filename, url))
        urlretrieve(url, filename)


HAARCASCADE_FRONTALFACE_DEFAULT = \
    'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
HAARCASCADE_EYE = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml'

HAARCASCADE_FRONTALFACE_ALT = \
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml"

HAARCASCADE_FRONTALFACE_ALT2 = \
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

HAARCASCADE_FRONTALFACE_ALT_TREE = \
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt_tree.xml"
