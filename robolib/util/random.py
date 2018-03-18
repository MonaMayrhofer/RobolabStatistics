import numpy as np


def random_different_numbers(max):
    ind1 = 0
    ind2 = 0
    while ind1 == ind2:
        ind1 = np.random.randint(max)
        ind2 = np.random.randint(max)
    return ind1, ind2
