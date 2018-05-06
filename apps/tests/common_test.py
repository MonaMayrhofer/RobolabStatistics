from robolib.networks.common import euclidean_distance
from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, BatchNormalization, Lambda
import tensorflow as tf
import keras.backend as back

a = tf.placeholder(tf.float32, shape=(None, 3))
b = tf.placeholder(tf.float32, shape=(None, 3))
sess = tf.Session()
back.set_session(sess)

distance = euclidean_distance([a, b])

dist_val = distance.eval(feed_dict={a: [[1, 0, 0]], b: [[0, 0, 1]]}, session=sess)

print(dist_val)
