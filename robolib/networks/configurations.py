
from keras.optimizers import SGD, RMSprop
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, BatchNormalization, Lambda, MaxPooling2D


class NetConfig:
    def create_base(self, input_d):
        pass

    def get_input_dim(self, input_image_size, input_to_output_stride, insets):
        pass

    def new_optimizer(self):
        pass


class ClassicConfig(NetConfig):
    def __init__(self):
        pass

    def create_base(self, input_d):
        print("Generating ClassicConfig")
        seq = Sequential()
        seq.add(Dense(200, activation='linear', input_shape=input_d))
        seq.add(Dense(100, activation='linear'))
        seq.add(Dropout(0.2))
        seq.add(Dense(50, activation='linear'))
        return seq

    def get_input_dim(self, input_image_size, input_to_output_stride, insets):
        return ((int(input_image_size[0] / input_to_output_stride) - insets[1] - insets[3]) *
                (int(input_image_size[1] / input_to_output_stride) - insets[0] - insets[2]),)

    def new_optimizer(self):
        return RMSprop()


class ConvolutionalConfig(NetConfig):
    def __init__(self):
        pass

    def create_base(self, input_d):
        print("Generating ConvolutionalConfig")
        seq = Sequential()
        seq.add(Conv2D(filters=9, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=input_d))
        seq.add(Flatten())
        seq.add(Dense(200, activation='linear'))
        seq.add(Dense(100, activation='linear'))
        seq.add(Dropout(0.2))
        seq.add(Dense(50, activation='linear'))
        return seq

    def get_input_dim(self, input_image_size, input_to_output_stride, insets):
        return (int(input_image_size[0] / input_to_output_stride) - insets[1] - insets[3],
                int(input_image_size[1] / input_to_output_stride) - insets[0] - insets[2], 1)

    def new_optimizer(self):
        return RMSprop()


class MultiConvConfig(NetConfig):
    def __init__(self):
        pass

    def create_base(self, input_d):
        print("Generating MultiConvConfig")
        seq = Sequential()
        seq.add(Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_d))
        seq.add(BatchNormalization())
        seq.add(Dropout(0.2))

        seq.add(Conv2D(8, (3, 3), activation='relu'))
        seq.add(BatchNormalization())
        seq.add(Dropout(0.2))

        seq.add(Conv2D(8, (3, 3), activation='relu'))
        seq.add(BatchNormalization())
        seq.add(Dropout(0.2))

        seq.add(Flatten())
        seq.add(Dense(500, activation='relu'))
        seq.add(Dropout(0.2))
        seq.add(Dense(500, activation='relu'))
        seq.add(Dropout(0.2))
        seq.add(Dense(50, activation='relu'))  # Why nan in loss when this is increased?
        return seq

    def get_input_dim(self, input_image_size, input_to_output_stride, insets):
        return (int(input_image_size[0] / input_to_output_stride) - insets[1] - insets[3],
                int(input_image_size[1] / input_to_output_stride) - insets[0] - insets[2], 1)

    def new_optimizer(self):
        return RMSprop()


class VGG19ish(NetConfig):
    def __init__(self):
        pass

    def create_base(self, input_d):
        print("Generating FaceNetInspired")
        seq = Sequential()
        seq.add(Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                       activation='relu', input_shape=input_d, name="conv1"))
        print("conv1 {0}".format(seq.output_shape))
        seq.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool1"))
        print("pool1 {0}".format(seq.output_shape))
        seq.add(BatchNormalization())

        seq.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv2"))
        print("conv2 {0}".format(seq.output_shape))
        seq.add(BatchNormalization())
        seq.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool2"))
        print("pool2 {0}".format(seq.output_shape))

        seq.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv3"))
        print("conv3 {0}".format(seq.output_shape))
        seq.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool3"))
        print("pool3 {0}".format(seq.output_shape))

        seq.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv4"))  # 4
        print("conv4 {0}".format(seq.output_shape))
        seq.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv5"))  # 5
        print("conv5 {0}".format(seq.output_shape))
        seq.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv6"))  # 6
        print("conv6 {0}".format(seq.output_shape))
        seq.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool4"))
        print("pool4 {0}".format(seq.output_shape))

        seq.add(Flatten(input_shape=(3, 4, 256), name='concat'))
        print("concat {0}".format(seq.output_shape))
        seq.add(Dense(4096, activation='relu', name="fc1"))
        print("fc1 {0}".format(seq.output_shape))
        seq.add(Dense(4096, activation='relu', name="fc2"))
        print("fc2 {0}".format(seq.output_shape))
        seq.add(Dense(128, activation='relu', name="fc3"))
        print("fc3 {0}".format(seq.output_shape))
        return seq

    def get_input_dim(self, input_image_size, input_to_output_stride, insets):
        return (int(input_image_size[0] / input_to_output_stride) - insets[1] - insets[3],
                int(input_image_size[1] / input_to_output_stride) - insets[0] - insets[2], 1)

    def new_optimizer(self):
        return SGD()  # LR = 0.01
