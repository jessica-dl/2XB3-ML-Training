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
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import gc


class CGANModel:

    def __init__(self, filepath, local=False, generator_weights=None, discriminator_weights=None, encoder_weights=None):
        self.filepath = filepath
        self.local = local
        self.generator_weights = generator_weights
        self.discriminator_weights = discriminator_weights
        self.encoder_weights = encoder_weights

        self.img_rows = 64
        self.img_cols = 64
        self.img_channels = 3  # 3
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.num_classes = 6  # 6
        self.latent_dim = 100

    def __build_gan(self):
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.__build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.__build_generator()

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
        self.encoder = self.__build_encoder()

        self.encoder.compile(loss='mse', optimizer=Adam(0.0001, 0.5))

    def __build_generator(self):
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

        if self.generator_weights is not None:
            self.__load_weights(model, self.generator_weights)

        return model

    def __build_discriminator(self):
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

        if self.discriminator_weights is not None:
            self.__load_weights(model, self.discriminator_weights)

        print('--- DISCRIMINATOR ---')
        model.summary()

        return model

    def __build_encoder(self):

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

        x = Flatten()(x)

        model = Model([input_image_x], x)

        if self.encoder_weights is not None:
            self.__load_weights(model, self.encoder_weights)

        print('--- ENCODER ---')
        model.summary()

        return model

    def __build_facenet(self):
        model = load_model('facenet_keras.h5')

        print('--- FACENET ---')
        model.summary()

        return model

    def train(self, dataset, log_dir):
        self.train_phase1(dataset)
        self.train_phase2(dataset)

    def train_gpu(self, dataset, log_dir):
        with tf.device('/device:GPU:0'):
            self.train(dataset, log_dir)

    def train_phase1(self, dataset, epochs=15000000, batch_size=32, sample_int=1000, cp_int=100000):
        self.__build_gan()
        print('Built GAN')

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs + 1):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, dataset.x.shape[0], batch_size)
            imgs, labels = dataset.x[idx], dataset.y[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

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
            sampled_labels = keras.utils.to_categorical(np.random.randint(0, self.num_classes, batch_size), self.num_classes)

            # Train the generator
            g_loss = self.gan.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_int == 0:
                self.sample_generator_images(epoch)

            if epoch % cp_int == 0 and epoch > 0:
                self.save_gan(epoch)

        self.save_gan()

    def train_phase2(self, dataset, epochs=100, train_samples=250, batch_size=32, sample_int=50, cp_int=200):
        # Clean up phase1 models to save VRAM
        self.discriminator = None
        gc.collect()

        self.__build_encoding()
        print('Built Encoder')

        # -----------------------
        # Create Fake Dataset
        # -----------------------
        print('Generating Encoder Dataset')
        x = None
        z = None
        while x is None or x.shape[0] < train_samples:
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            sampled_labels = keras.utils.to_categorical(np.random.randint(0, self.num_classes, batch_size), self.num_classes)

            next_batch = self.generator.predict([noise, sampled_labels])
            x = next_batch if x is None else np.concatenate([x, next_batch])
            z = noise if z is None else np.concatenate([z, noise])
            print("[%d/%d]" % (x.shape[0], train_samples))

        for epoch in range(epochs + 1):
            idx = np.random.randint(0, x.shape[0], batch_size)
            x_batch, z_batch = x[idx], z[idx]

            e_loss = self.encoder.train_on_batch(x_batch, z_batch)

            # Plot the progress
            print("%d [E loss: %f]" % (epoch, e_loss))

            # If at save interval => save generated image samples
            if epoch % sample_int == 0:
                self.sample_encoder_images(dataset, epoch)

            if epoch % cp_int == 0 and epoch > 0:
                self.save_encoding(epoch)

        self.save_encoding()

    def sample_generator_images(self, epoch):
        r, c = 2, self.num_classes // 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = keras.utils.to_categorical(np.arange(0, self.num_classes), self.num_classes)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, ...])
                axs[i, j].set_title("Label: %d" % np.argmax(sampled_labels[cnt]))
                axs[i, j].axis('off')
                cnt += 1

        filename = "G%s.png" % str(epoch).zfill(8)
        fig.savefig(filename)
        plt.close()

        # Local dirs are not automatically created
        if self.local:
            try:
                os.makedirs(self.filepath + 'samples/')
            except FileExistsError:
                pass

        with file_io.FileIO(filename, mode='rb') as input_f:
            with file_io.FileIO(self.filepath + 'samples/' + filename, mode='wb+') as output_f:
                output_f.write(input_f.read())

        # Clean up local file
        os.remove(filename)

    def sample_encoder_images(self, dataset, epoch):
        r, c = 4, 6

        # Select a random set of images
        idx = np.random.randint(0, dataset.x.shape[0], (r * c) // 2)
        imgs, labels = dataset.x[idx], dataset.y[idx]

        # Encode and regen images
        z = self.encoder.predict(imgs)
        gen_imgs = self.generator.predict([z, labels])

        # Rescale images from [-1, 1] to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if cnt % 2 == 0:
                    axs[i, j].imshow(imgs[(cnt // 2), ...])
                    axs[i, j].set_title("O: %d" % np.argmax(labels[cnt // 2]))
                    axs[i, j].axis('off')
                else:
                    axs[i, j].imshow(gen_imgs[(cnt // 2), ...])
                    axs[i, j].set_title("G: %d" % np.argmax(labels[cnt // 2]))
                    axs[i, j].axis('off')
                cnt += 1

        filename = "E%s.png" % str(epoch).zfill(8)
        fig.savefig(filename)
        plt.close()

        # Local dirs are not automatically created
        if self.local:
            try:
                os.makedirs(self.filepath + 'samples/')
            except FileExistsError:
                pass

        with file_io.FileIO(filename, mode='rb') as input_f:
            with file_io.FileIO(self.filepath + 'samples/' + filename, mode='wb+') as output_f:
                output_f.write(input_f.read())

        # Clean up local file
        os.remove(filename)

    def save_gan(self, epoch=None):
        self.save_model(self.generator, 'generator' + ('-' + str(epoch) if epoch else ''))
        self.save_model(self.discriminator, 'discriminator' + ('-' + str(epoch) if epoch else ''))

    def save_encoding(self, epoch=None):
        self.save_model(self.encoder, 'encoder' + ('-' + str(epoch) if epoch else ''))

    def save_model(self, model, name):
        filename = name + '.h5'

        # Local dirs are not automatically created
        if self.local:
            try:
                os.makedirs(self.filepath + 'models/')
            except FileExistsError:
                pass

        model.save(filename)
        with file_io.FileIO(filename, mode='rb') as input_f:
            with file_io.FileIO(self.filepath + 'models/' + filename, mode='wb+') as output_f:
                output_f.write(input_f.read())

        # Clean up local file
        os.remove(filename)

    def __load_weights(self, model, filepath):
        if not self.local:
            with file_io.FileIO(filepath, mode='rb') as weights_f:
                with open('weights.h5', 'wb') as local_weights:
                    local_weights.write(weights_f.read())
            filepath = 'weights.h5'

        model.load_weights(filepath)










