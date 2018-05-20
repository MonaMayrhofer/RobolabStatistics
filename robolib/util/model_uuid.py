import time


def get_model_identifier(epochs: int):
    t = int(time.time())
    t *= 1000
    t += int(epochs/100) % 10000
    return t
