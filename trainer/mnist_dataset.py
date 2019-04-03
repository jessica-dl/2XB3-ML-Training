import numpy as np
import keras
from keras.datasets import mnist


class Dataset:

    def __init__(self, path, local):
        self.path = path
        self.local = local

        (x, y), (_, _) = mnist.load_data()

        # Configure input
        x = (x.astype(np.float32) - 127.5) / 127.5
        x = np.expand_dims(x, axis=3)
        x_padding = np.zeros((x.shape[0], 64, 64, 1)) - 1
        x_padding[:, :28, :28, :] = x
        x = x_padding
        y = keras.utils.np_utils.to_categorical(y, 10)

        self.x = x
        self.y = y

        print('Loaded dataset')
        print('X:', self.x.shape)
        print('Y:', self.y.shape)






