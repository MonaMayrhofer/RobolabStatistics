from keras import backend


def contrastive_loss(y_true, y_pred):
    margin = 1
    return backend.mean(
        y_true * backend.square(y_pred) + (1 - y_true) * backend.square(backend.maximum(margin - y_pred, 0)))


def euclidean_distance(vectors):
    x, y = vectors
    return backend.sqrt(backend.sum(backend.square(x - y), axis=1, keepdims=True))


def euclidean_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1
