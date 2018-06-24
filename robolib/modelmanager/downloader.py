import os.path
from urllib.request import urlretrieve
import robolib.datamanager.datadir as datadir


def get_model(model, delete=False):
    filename = datadir.get_opencv_dir(model["name"])
    if delete and os.path.isfile(filename):
        print("Delete-Mode is on. Deleting file")
        os.remove(filename)
    if os.path.isfile(filename):
        print("{} already present, to re-download it remove the file.".format(filename))
    else:
        print("{} not found, downloading it from: {} ".format(filename, model["url"]))
        urlretrieve(model["url"], filename)
    return filename


HAARCASCADE_FRONTALFACE_DEFAULT = \
    {"url": 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default'
            '.xml',
     "name": 'Frontalface.xml'}
HAARCASCADE_EYE = \
    {"url": 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml',
     "name": 'Eye.xml'}

HAARCASCADE_FRONTALFACE_ALT = \
    {"url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml",
     "name": "Frontalface_alt.xml"}

HAARCASCADE_FRONTALFACE_ALT2 = \
    {"url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml",
     "name": "Frontalface_alt2.xml"}

HAARCASCADE_FRONTALFACE_ALT_TREE = \
    {"url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades"
            "/haarcascade_frontalface_alt_tree.xml",
     "name": "Frontalface_alt_tree.xml"}
