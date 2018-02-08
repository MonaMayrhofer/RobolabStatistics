import tensorflow as tf
import numpy as np
from random import randint


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


def model_function(features, labels, mode):

    flat = tf.reshape(features, [4, 1])

    class_a = tf.layers.dense(inputs=flat, units=4, name="dense_A_layer", activation=tf.nn.relu)
    class_b = tf.layers.dense(inputs=class_a, units=2, name="dense_B_layer", activation=tf.nn.relu)
    collector = tf.layers.dense(inputs=class_b, units=1, name="collector_layer")

    final_prediction = tf.argmax(collector, axis=1, name="prediction_tensor")

    if mode == tf.estimator.ModeKeys.PREDICT:
        return final_prediction

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=collector)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss)
        return loss, train_op

    print("Not a known mode")
    return None


f_features = tf.placeholder(tf.float32, (4,), name="features_placeholder")
f_labels = tf.placeholder(tf.int32, (), name="labels_placeholder")

train_loss, train_model = model_function(f_features, f_labels, tf.estimator.ModeKeys.TRAIN)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(1000):
        x_value = np.random.rand()
        y_value = x_value * 2 + 6

        feats, lab = generate_inputs()

        session.run(train_model, feed_dict={f_features: feats, f_labels: lab})
        t_loss = session.run(train_loss, feed_dict={f_features: feats, f_labels: lab})
        print(t_loss)
