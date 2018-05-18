import time


def get_model_identifier(epochs: int):
    t = int(time.time())
    print(t)
    t *= 1000
    print(t)
    t += int(epochs/100) % 10000
    print(t)
    return t
