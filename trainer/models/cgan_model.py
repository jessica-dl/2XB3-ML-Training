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
import matplotlib.pyplot as plt
import tensorflow as tf


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
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.gan = Model([noise, label], valid)
        self.gan.compile(loss=['binary_crossentropy'],
                         optimizer=optimizer)

    def __build_encoding(self):
        self.encoder = self.build_encoder()

        self.encoder.compile(loss='crossentropy', optimizer=Adam(0.0001, 0.5))

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

    def build_discriminator(self):
        input_image_x = Input(self.img_shape)
        input_labels_y = Input(shape=(self.num_classes,))

        # Replicate y
        labels_y = Reshape(target_shape=(1, 1, -1))(input_labels_y)
        labels_y = Lambda(K.tile, arguments={'n': (1, self.img_rows // 2, self.img_rows // 2, self.num_classes)})(
            labels_y)

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

    def train(self, dataset, log_dir):
        self.train_phase1()

    def train_gpu(self, dataset, log_dir):
        with tf.device('/device:GPU:0'):
            self.train(dataset, log_dir)

    def train_phase1(self, epochs=2000, batch_size=128, sample_interval=100):
        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        x_padding = np.zeros((60000, 64, 64, 1)) - 1
        x_padding[:, :28, :28, :] = X_train
        X_train = x_padding
        y_train = keras.utils.np_utils.to_categorical(y_train, 10)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = keras.utils.to_categorical(np.random.randint(0, 10, batch_size), 10)

            # Train the generator
            g_loss = self.gan.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = keras.utils.to_categorical(np.arange(0, 10), 10)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].set_title("Digit: %d" % np.argmax(sampled_labels[cnt]))
                axs[i, j].axis('off')
                cnt += 1

        filename = "%d.png" % epoch
        fig.savefig(filename)
        plt.close()

        with file_io.FileIO(filename, mode='rb') as input_f:
            with file_io.FileIO(self.filepath + 'samples/' + filename, mode='wb+') as output_f:
                output_f.write(input_f.read())

    def train_phase2(self):
        pass

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






