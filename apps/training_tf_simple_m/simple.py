import tensorflow as tf
from random import randint
import matplot


def generate_inputs():
    data = [
        0.0, 0.0,
        0.0, 0.0
    ]
    label = randint(0, 1)

    if label == 0:
        data[0] = 1.0
        data[1] = 1.0
    elif label == 1:
        data[0] = 1.0
        data[2] = 1.0

    return data, label


def create_inputs():
    data, label = generate_inputs()

    feature_cols = tf.constant(data)
    labels = tf.constant(label)
    return feature_cols, labels


def create_inputs_from(data):
    if len(data) == 0:
        raise StopIteration
    dat = data.pop(0)
    return tf.constant(dat)

def create_inputs_old():
    print("Creating inputs...")
    filenames = ["/tmp/file1.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        print("Parsing...")
        keys_to_features = {
            "data": tf.FixedLenFeature((4, ), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
            "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        data = tf.cast(parsed["data"], tf.float32)
        label = tf.cast(parsed["label"], tf.int64)
        print(data)
        print("Parsed!")

        return data, label

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    print("Created inputs!")
    return features, labels


def model_function(features, labels, mode):

    flat = tf.reshape(features, [4, 1])

    class_a = tf.layers.dense(inputs=flat, units=4, name="dense_A_layer", activation=tf.nn.relu)
    class_b = tf.layers.dense(inputs=class_a, units=2, name="dense_B_layer", activation=tf.nn.relu)
    collector = tf.layers.dense(inputs=class_b, units=1, name="collector_layer")

    final_prediction = tf.argmax(collector, axis=1, name="prediction_tensor")

    if mode == tf.estimator.ModeKeys.PREDICT:
        print("PredictModel")
        print("Return")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=final_prediction)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=collector)

    if mode == tf.estimator.ModeKeys.TRAIN:
        print("TrainModel")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        print("Return")
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        print("EvalModel")
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=final_prediction)
        }
        print("Return")
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    print("Not a known mode")
    return None


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    estimator = tf.estimator.Estimator(model_fn=model_function, model_dir="tmp/simplemodel")

    print("Training...")
    estimator.train(input_fn=create_inputs, hooks=[], steps=200000)
    print("Trained!")

    while True:
        print("Predicting...")

        data, expected = generate_inputs()

        data_list = [data]

        predictions = estimator.predict(input_fn=lambda: create_inputs_from(data_list))
        for predict in predictions:
            print("{}, expected: {}".format(predict, expected))
            if predict != expected:
                return
            break


if __name__ == "__main__":
    main()
