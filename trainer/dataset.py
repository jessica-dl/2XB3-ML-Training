from tensorflow.python.lib.io import file_io
import h5py
import numpy as np


class Dataset:

    def __init__(self, path, local):
        self.path = path
        self.local = local

        if not local:
            with file_io.FileIO(path, mode='rb') as dataset_f:
                with open('dataset.h5', 'wb') as local_dataset:
                    local_dataset.write(dataset_f.read())
            path = 'dataset.h5'

            hf = h5py.File(path, 'r')
            self.x = hf.get('x')[:]
            self.y = hf.get('y')[:]
            hf.close()

            self.x = (self.x.astype(np.float32) - 127.5) / 127.5

            self.__make_overfit()

            print('Loaded dataset')
            print('X:', self.x.shape)
            print('Y:', self.y.shape)

    def __make_overfit(self):
        """
        Modify dataset for overfitting by only including 3 samples from each class
        :return:
        """
        minimal_x = self.x[:1]
        minimal_y = self.y[:1]

        per_class = 3

        i = 1
        found = np.array([0 for _ in range(self.y.shape[-1])])
        found[np.argmax(minimal_y[0])] += 1

        while sum(found) < self.y.shape[-1] * per_class:
            for c in range(self.y.shape[-1]):
                if found[np.argmax(self.y[i])] < per_class:
                    minimal_x = np.concatenate([minimal_x, self.x[i:i+1]])
                    minimal_y = np.concatenate([minimal_y, self.y[i:i+1]])
                    found[np.argmax(self.y[i])] += 1
                i += 1

        self.x = minimal_x
        self.y = minimal_y





