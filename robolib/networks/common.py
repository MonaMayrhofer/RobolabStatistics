from keras import backend


def contrastive_loss(y_true, y_pred):
    margin = 5
    return backend.mean(
         y_true * backend.square(y_pred) +
         (1 - y_true) * backend.square(backend.maximum(margin - y_pred, (y_pred-margin)/10)))


def contrastive_loss_manual(correct, prediction):
    if correct:
        return prediction**2
    else:
        return max(100-prediction, (prediction-100)/10)**2


def euclidean_distance(vectors):
    x, y = vectors
    return backend.sqrt(backend.sum(backend.square(x - y), axis=1, keepdims=True))


def euclidean_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1
