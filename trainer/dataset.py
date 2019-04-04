from tensorflow.python.lib.io import file_io
import h5py


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

            print('Loaded dataset')
            print('X:', self.x.shape)
            print('Y:', self.y.shape)







