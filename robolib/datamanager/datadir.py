import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.join(os.path.abspath(__file__))), '..', '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PAPARAZZI_DIR = os.path.join(DATA_DIR, 'paparazzi')


def paparazzi_output(name: str, group: str = 'Images'):
    return os.path.join(PAPARAZZI_DIR, group, name)
