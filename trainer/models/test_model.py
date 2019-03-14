from trainer.models.aging_model import AgingModel
import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.layers import Dropout, MaxPooling2D
from keras import callbacks
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import numpy as np


class TestModel(AgingModel):

    def __init__(self):
        super(TestModel, self).__init__()

        # Loading the data
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.train_data = mnist.train.images  # Returns np.array
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.eval_data = mnist.test.images  # Returns np.array
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        # Pre processing the data
        self.train_labels = keras.utils.np_utils.to_categorical(self.train_labels, 10)
        self.eval_labels = keras.utils.np_utils.to_categorical(self.eval_labels, 10)
        self.train_data = np.reshape(self.train_data, [-1, 28, 28, 1])
        self.eval_data = np.reshape(self.eval_data, [-1, 28, 28, 1])

        # Initializing the model
        self.model = self.model_def(self.train_data.shape[1:])

        # Compiling the model
        self.model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Printing the model summary
        self.model.summary()

    def train(self, dataset, log_dir):
        with tf.device('/device:GPU:0'):
            # Adding the callback for TensorBoard and History
            tensorboard = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                                                write_images=True)

            # Training the model
            self.model.fit(x=self.train_data,
                           y=self.train_labels,
                           epochs=1,
                           verbose=1,
                           batch_size=100,
                           callbacks=[tensorboard],
                           validation_data=(self.eval_data, self.eval_labels))

    def save(self, filepath):
        # Save model.h5 on to google storage
        self.model.save('model.h5')
        with file_io.FileIO('model.h5', mode='rb') as input_f:
            with file_io.FileIO(filepath, mode='wb+') as output_f:
                output_f.write(input_f.read())

    def model_def(self, input_shape):
        # First input
        X_input = Input(input_shape)

        # Convolutional Layer 1
        X = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', name='conv1')(X_input)
        X = Activation('relu')(X)

        # Max pooling layer 1
        X = MaxPooling2D(pool_size=(2, 2), strides=2, name='maxpool1')(X)

        # Convolutional Layer 2
        X = Conv2D(filters=64, kernel_size=[5, 5], padding='same', name='conv2')(X)
        X = Activation('relu')(X)

        # Max Pooling Layer 2
        X = MaxPooling2D(pool_size=(2, 2), strides=2, name='maxpool2')(X)

        # Flatten
        X = Flatten()(X)

        # Dense Layer
        X = Dense(1024, activation='relu', name='dense_1')(X)

        # Dropout layer
        X = Dropout(0.4, name='dropout')(X)

        # dense 2 layer
        X = Dense(10, activation='softmax', name='dense_2')(X)

        # The model object
        model = Model(inputs=X_input, outputs=X, name='cnnMINSTModel')

        return model

