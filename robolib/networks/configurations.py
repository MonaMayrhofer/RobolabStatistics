from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, BatchNormalization, Lambda, MaxPooling2D, Softmax, ZeroPadding2D


class NetConfig:
    def create_base(self, input_d):
        pass

    def get_input_dim(self, input_image_size, input_to_output_stride, insets):
        pass

    def new_optimizer(self):
        pass

    def get_input_to_output_stride(self):
        pass

    def debug_output(self, name, shape):
        print("{0: >25} {1}".format(name, shape))

    def add(self, seq, layer):
        seq.add(layer)
        self.debug_output(layer.name, seq.output_shape)


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

    def get_input_to_output_stride(self):
        return 2

    def get_input_image_size(self):
        return 96, 128


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

    def get_input_to_output_stride(self):
        return 2

    def get_input_image_size(self):
        return 96, 128


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

    def get_input_to_output_stride(self):
        return 2

    def get_input_image_size(self):
        return 96, 128


class VGG19ish(NetConfig):
    def __init__(self):
        pass

    def create_base(self, input_d):
        print("Generating VGG19ish")
        seq = Sequential()

        def a(layer):
            self.add(seq, layer)

        a(Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', input_shape=input_d, name="conv1_1"))
        a(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool1"))
        a(BatchNormalization())
        a(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv2_1"))
        a(BatchNormalization())
        a(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool2"))
        a(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv3_1"))
        a(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv3_2"))
        a(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool3"))
        a(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv4_1"))
        a(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv4_2"))
        a(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool4"))
        a(Flatten(input_shape=(3, 4, 256), name='concat'))
        a(Dense(4096, activation='relu', name="fc1"))
        a(Dense(4096, activation='relu', name="fc2"))
        a(Dense(128, activation='relu', name="fc3"))
        return seq

    def get_input_dim(self, input_image_size, input_to_output_stride, insets):
        return (int(input_image_size[0] / input_to_output_stride) - insets[1] - insets[3],
                int(input_image_size[1] / input_to_output_stride) - insets[0] - insets[2], 1)

    def new_optimizer(self):
        return SGD()  # LR = 0.01

    def get_input_to_output_stride(self):
        return 2

    def get_input_image_size(self):
        return 96, 128


class VGG19advanced(NetConfig):
    def __init__(self):
        pass

    def create_base(self, input_d):
        print("Generating VGG19simplified")
        self.debug_output("input", input_d)

        seq = Sequential()
        self.add(seq, Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv1_1", input_shape=input_d))
        self.add(seq, Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv1_2"))
        self.add(seq, MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool1"))
        self.add(seq, Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv2_1"))
        self.add(seq, Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv2_2"))
        self.add(seq, MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool2"))
        self.add(seq, Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv3_1"))
        self.add(seq, Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv3_2"))
        # self.add(seq, Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv3_3"))
        self.add(seq, MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool3"))
        self.add(seq, Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv4_1"))
        self.add(seq, Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv4_2"))
        # self.add(seq, Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv4_3"))
        self.add(seq, MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool4"))
        self.add(seq, Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv5_1"))
        self.add(seq, Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv5_2"))
        # self.add(seq, Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv5_3"))
        self.add(seq, MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool5"))

        self.add(seq, Flatten(name='concat'))
        self.add(seq, Dense(4096, activation='relu', name="fc6"))
        self.add(seq, Dense(4096, activation='relu', name="fc7"))
        self.add(seq, Dropout(0.5))
        self.add(seq, Dense(128, activation='relu', name="fc8"))
        self.add(seq, Softmax(name="prob"))
        return seq

    def get_input_dim(self, input_image_size, input_to_output_stride, insets):
        return (int(input_image_size[0] / input_to_output_stride) - insets[1] - insets[3],
                int(input_image_size[1] / input_to_output_stride) - insets[0] - insets[2], 1)

    def new_optimizer(self):
        return SGD(lr=0.001)  # LR = 0.01

    def get_input_to_output_stride(self):
        return 2

    def get_input_image_size(self):
        return 96, 128


class VGG19pretrained(NetConfig):
    def __init__(self):
        pass

    def create_base(self, input_d):
        print("Generating VGG19simplified")
        self.debug_output("input", input_d)

        seq = Sequential()
        self.add(seq, ZeroPadding2D(padding=(1, 1), input_shape=input_d))
        self.add(seq, Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv1_1'))
        self.add(seq, ZeroPadding2D(padding=(1, 1)))
        self.add(seq, Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv1_2'))
        self.add(seq, MaxPooling2D((2, 2), strides=(2, 2)))

        self.add(seq, ZeroPadding2D((1, 1)))
        self.add(seq, Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv2_1'))
        self.add(seq, ZeroPadding2D((1, 1)))
        self.add(seq, Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv2_2'))
        self.add(seq, MaxPooling2D((2, 2), strides=(2, 2)))

        self.add(seq, ZeroPadding2D((1, 1)))
        self.add(seq, Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv3_1'))
        self.add(seq, ZeroPadding2D((1, 1)))
        self.add(seq, Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv3_2'))
        self.add(seq, ZeroPadding2D((1, 1)))
        self.add(seq, Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv3_3'))
        self.add(seq, MaxPooling2D((2, 2), strides=(2, 2)))

        self.add(seq, ZeroPadding2D((1, 1)))
        self.add(seq, Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv4_1'))
        self.add(seq, ZeroPadding2D((1, 1)))
        self.add(seq, Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv4_2'))
        self.add(seq, ZeroPadding2D((1, 1)))
        self.add(seq, Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv4_3'))
        self.add(seq, MaxPooling2D((2, 2), strides=(2, 2)))

        self.add(seq, Flatten())
        self.add(seq, Dense(4096, activation='relu', name='fc6'))
        self.add(seq, Dropout(0.5))
        self.add(seq, Dense(4096, activation='relu', name='fc7'))
        self.add(seq, Dropout(0.5))
        self.add(seq, Dense(128, activation='softmax', name='fc8'))

        return seq

    def get_input_image_size(self):
        return 96, 128

    def get_input_dim(self, input_image_size, input_to_output_stride, insets):
        return (int(input_image_size[0] / input_to_output_stride) - insets[1] - insets[3],
                int(input_image_size[1] / input_to_output_stride) - insets[0] - insets[2], 1)

    def new_optimizer(self):
        return SGD(lr=0.0001)  # LR = 0.01

    def get_input_to_output_stride(self):
        return 2
