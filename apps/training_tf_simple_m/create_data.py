import tensorflow as tf
from random import randint
import sys


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _lists_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


train_filename = "/tmp/file1.tfrecord"

writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(0, 1000):
    if i % 100 == 0:
        print("Done: %d" % i)

    label = randint(0, 3)
    data = [0, 0, 0, 0]
    data[label] = 1

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'data': _lists_feature(data),
                'label': _int64_feature(label)
            }
        ))

    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()
