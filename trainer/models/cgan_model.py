from trainer.models.aging_model import AgingModel
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Concatenate, BatchNormalization, Lambda, Activation, Reshape, Conv2D, Deconv2D, ReLU, LeakyReLU
from keras import callbacks
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import numpy as np


class CGANModel(AgingModel):

    def __init__(self):
        super(CGANModel, self).__init__()
        self.img_rows = 64
        self.img_cols = 64
        self.img_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.num_classes = 10
        self.latent_dim = 100

        self.generator = self.build_generator()
        print(self.generator)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
        # Discriminator is not trainable in the combined model
        self.discriminator.trainable = False

        input_z = Input(shape=(self.latent_dim,))
        input_y = Input(shape=(self.num_classes,))

        output_x = self.generator([input_z, input_y])
        output_o = self.discriminator([output_x, input_y])

        self.gan = Model([input_z, input_y], output_o)

        print('--- GAN ---')
        self.gan.summary()

        self.gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    def train(self, dataset, log_dir):
        pass

    def save(self, filepath):
        pass

    def build_discriminator(self):
        input_image_x = Input(self.img_shape)
        input_labels_y = Input(shape=(self.num_classes,))

        # Replicate y
        labels_y = Reshape(target_shape=(1, 1, -1))(input_labels_y)
        labels_y = Lambda(K.tile, arguments={'n': (-1, self.img_rows // 2, self.img_rows // 2, self.num_classes)})(labels_y)

        # Conv 1
        x = Conv2D(kernel_size=(4, 4), strides=(2, 2), filters=64, padding='same')(input_image_x)
        x = LeakyReLU()(x)

        # Concat y
        x = Concatenate()([x, labels_y])

        # Conv 2
        x = Conv2D(kernel_size=(4, 4), strides=(2, 2), filters=128, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Conv 3
        x = Conv2D(kernel_size=(4, 4), strides=(2, 2), filters=256, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Conv 4
        x = Conv2D(kernel_size=(4, 4), strides=(2, 2), filters=512, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Conv 5
        x = Conv2D(kernel_size=(4, 4), strides=(1, 1), filters=1)(x)
        x = Activation(activation='sigmoid')(x)

        model = Model([input_image_x, input_labels_y], x)

        print('--- DISCRIMINATOR ---')
        model.summary()

        return model

    def build_generator(self):
        # Input
        latent_z = Input(shape=(self.latent_dim,))
        label_y = Input(shape=(self.num_classes,))
        x = Concatenate()([latent_z, label_y])
        x = Reshape(target_shape=(1, 1, -1))(x)

        # Full Conv 1
        x = Deconv2D(kernel_size=[4, 4], strides=(2, 2), filters=512)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Full Conv 2
        x = Deconv2D(kernel_size=[4, 4], strides=(2, 2), filters=256, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Full Conv 3
        x = Deconv2D(kernel_size=[4, 4], strides=(2, 2), filters=128, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        # Full Conv 4
        x = Deconv2D(kernel_size=[4, 4], strides=(2, 2), filters=64, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Full Conv 5
        x = Deconv2D(kernel_size=[4, 4], strides=(2, 2), filters=self.img_channels, padding='same')(x)
        x = Activation(activation='tanh')(x)

        model = Model([latent_z, label_y], x)

        print('--- GENERATOR ---')
        model.summary()

        return model





