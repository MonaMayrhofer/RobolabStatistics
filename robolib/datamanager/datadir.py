import os

__ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.join(os.path.abspath(__file__))), '..', '..'))
__DATA_DIR = os.path.join(__ROOT_DIR, 'data')
__MODEL_DIR = os.path.join(__DATA_DIR, 'models')
__CONVERTED_IMG_DIR = os.path.join(__DATA_DIR, 'converted')


def converted_image_dir():
    return __CONVERTED_IMG_DIR


def paparazzi_output(name: str, group: str = 'Images'):
    return os.path.join(converted_image_dir(), group, name)


def get_model_dir(name):
    return os.path.join(__MODEL_DIR, name);