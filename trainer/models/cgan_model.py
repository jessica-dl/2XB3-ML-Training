from trainer.models.aging_model import AgingModel
import keras
from keras.models import Model
from keras.layers import Input, Concatenate, BatchNormalization, Lambda, Activation, Reshape, Conv2D, Deconv2D, ReLU, \
    LeakyReLU, Dense, Flatten, Dropout
from keras.datasets import mnist
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K
from tensorflow.python.lib.io import file_io
import numpy as np
import cv2


class CGANModel(AgingModel):

    def __init__(self, filepath):
        super(CGANModel, self).__init__()
        self.filepath = filepath
        self.img_rows = 64
        self.img_cols = 64
        self.img_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.num_classes = 10  # 6
        self.latent_dim = 100

        self.__build_gan()
        self.__build_encoding()

    def __build_gan(self):
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        self.generator = self.build_generator()

        # Discriminator is not trainable in the combined model
        self.discriminator.trainable = False

        input_z = Input(shape=(self.latent_dim,))
        input_y = Input(shape=(self.num_classes,))

        output_x = self.generator([input_z, input_y])
        output_o = self.discriminator([output_x, input_y])

        self.gan = Model([input_z, input_y], output_o)

        print('--- GAN ---')
        self.gan.summary()

        self.gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))

    def __build_encoding(self):
        self.encoder = self.build_encoder()
        self.generator = self.build_generator()
        self.generator.trainable = False

    def train(self, dataset, log_dir):
        self.train_phase1()

    def train_phase1(self, epochs=200, batch_size=32):
        (x_train, y_train), (_, _) = mnist.load_data()

        # Configure input
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = np.expand_dims(x_train, axis=3)
        x_padding = np.zeros((60000, 64, 64, 1)) - 1
        x_padding[:, :28, :28, :] = x_train
        x_train = x_padding
        y_train = keras.utils.np_utils.to_categorical(y_train, 10)

        print(x_train[0])
        # cv2.imshow('test', np.dstack(3*[((x_train[0] + 1) * 255).astype(np.uint8)]))
        # cv2.waitKey(0)
        print(y_train[0])

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images with list of random indices
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs, labels = x_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            cv2.imshow('test', np.dstack(3*[((gen_imgs[0] + 1) * 255).astype(np.uint8)]))
            cv2.waitKey(100)

            # Train the discriminator
            # print('real:', self.discriminator.predict([imgs, labels])[:5])
            # print('fake', self.discriminator.predict([gen_imgs, labels])[:5])
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = keras.utils.to_categorical(np.random.randint(0, 10, batch_size), 10)

            # Train the generator
            # g_out = self.gan.predict([noise, sampled_labels])
            # print('g_out', g_out[:5])
            g_loss = self.gan.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    def train_phase2(self):
        pass

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

    def generator_samples(self, num_samples=3):
        for c in range(self.num_classes):
            for i in range(num_samples):
                noise = np.random.normal(0, 1, (1, self.latent_dim))
                label = np.zeros((1, self.num_classes))
                label[0][c] = 1
                output_img = self.generator.predict([noise, label])[0]

                filename = 'sample' + str(i) + '-' + str(c) + '.png'
                cv2.imwrite(filename, np.dstack(3*[((output_img + 1) * 255).astype(np.uint8)]))

                with file_io.FileIO(filename, mode='rb') as input_f:
                    with file_io.FileIO(self.filepath + 'samples/' + filename, mode='wb+') as output_f:
                        output_f.write(input_f.read())

    def build_discriminator(self):
        input_image_x = Input(self.img_shape)
        input_labels_y = Input(shape=(self.num_classes,))

        # Replicate y
        labels_y = Reshape(target_shape=(1, 1, -1))(input_labels_y)
        labels_y = Lambda(K.tile, arguments={'n': (1, self.img_rows // 2, self.img_rows // 2, self.num_classes)})(labels_y)

        # Conv 1
        x = Conv2D(kernel_size=(4, 4), strides=(2, 2), filters=64, padding='same')(input_image_x)
        x = Dropout(0.4)(x)
        x = LeakyReLU()(x)

        # Concat y
        x = Concatenate()([x, labels_y])

        # Conv 2
        x = Conv2D(kernel_size=(4, 4), strides=(2, 2), filters=128, padding='same')(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Conv 3
        x = Conv2D(kernel_size=(4, 4), strides=(2, 2), filters=256, padding='same')(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Conv 4
        x = Conv2D(kernel_size=(4, 4), strides=(2, 2), filters=512, padding='same')(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Conv 5
        x = Conv2D(kernel_size=(4, 4), strides=(1, 1), filters=1)(x)
        x = Activation(activation='sigmoid')(x)

        x = Flatten()(x)

        model = Model([input_image_x, input_labels_y], x)

        print('--- DISCRIMINATOR ---')
        model.summary()

        return model

    def build_encoder(self):

        input_image_x = Input(self.img_shape)

        # Conv 1
        x = Conv2D(kernel_size=(5, 5), strides=(2, 2), filters=32)(input_image_x)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        # Conv 2
        x = Conv2D(kernel_size=(5, 5), strides=(2, 2), filters=64)(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        # Conv 3
        x = Conv2D(kernel_size=(5, 5), strides=(2, 2), filters=128)(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        # Conv 4
        x = Conv2D(kernel_size=(5, 5), strides=(2, 2), filters=256)(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        # FC 1
        x = Dense(4096)(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        # FC 2
        x = Dense(self.latent_dim)(x)

        model = Model([input_image_x], x)

        print('--- ENCODER ---')
        model.summary()

        return model

    def build_facenet(self):
        model = load_model('facenet_keras.h5')

        print('--- FACENET ---')
        model.summary()

        return model

    def save(self):
        self.save_model(self.generator, 'generator')
        self.save_model(self.discriminator, 'discriminator')
        self.save_model(self.encoder, 'encoder')

    def save_model(self, model, name):
        filename = name + '.h5'

        model.save(filename)
        with file_io.FileIO(filename, mode='rb') as input_f:
            with file_io.FileIO(self.filepath + 'models/' + filename, mode='wb+') as output_f:
                output_f.write(input_f.read())






