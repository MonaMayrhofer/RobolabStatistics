import os.path


def list_dir_recursive(path):
    assert os.path.exists(path), "Cannot traverse non-existent path"
    if not os.path.isdir(path):
        yield path
    else:
        for file in os.listdir(path):
            yield from list_dir_recursive(os.path.join(path, file))
