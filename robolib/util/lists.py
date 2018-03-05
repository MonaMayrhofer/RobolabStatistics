

def reverse_enumerate(list):
    """Enumerates a list in reverse, with index"""
    n = len(list)
    for obj in reversed(list):
        n -= 1
        yield n, obj;


def columns(mat):
    for _, col in enumerate_columns(mat):
        yield col


def enumerate_columns(mat):
    for col in range(mat.shape[1]):
        yield col, mat[:, col]


def reverse_enumerate_columns(mat):
    n = mat.shape[1]-1
    for col in range(n):
        yield n-col, mat[:, n-col]
