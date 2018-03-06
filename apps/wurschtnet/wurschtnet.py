from keras import backend
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Lambda


def load_wurscht_model(name):
    return load_model(name, custom_objects={'contrastive_loss': contrastive_loss, 'backend': backend})


def euclidean_distance(vectors):
    x, y = vectors
    return backend.sqrt(backend.sum(backend.square(x - y), axis=1, keepdims=True))


def euclidean_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    margin = 1
    return backend.mean(y_true * backend.square(y_pred) + (1 - y_true) * backend.square(backend.maximum(margin - y_pred, 0)))


def create_wurschtnet_base(input_d, hidden_layer_size):
    seq = Sequential()
    for i in range(len(hidden_layer_size)):
        if i == 0:
            seq.add(Dense(hidden_layer_size[i], input_shape=(input_d,), activation='linear'))
        else:
            seq.add(Dense(hidden_layer_size[i], activation='linear'))
        seq.add(Dropout(0.2))
    return seq


def create_wurschtnet(input_dim):
    input_size = input_dim[0]*input_dim[1]
    hidden_layer_sizes = [200, 100, 50]
    input_a = Input(shape=(input_size,))
    input_b = Input(shape=(input_size,))
    base_network = create_wurschtnet_base(input_size, hidden_layer_sizes)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=euclidean_dist_output_shape)([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model
