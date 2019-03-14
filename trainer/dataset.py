from tensorflow.python.lib.io import file_io


class Dataset:

    def __init__(self, url, data):
        """
        Initialize Dataset (will create it if it does not exist)
        :param url: The path to the dataset in GC buckets
        :param data: List of dicts of {"url": string, "age": number}
        """
        self.url = url
        self.data = data

        self.__setup()

    def __setup(self):
        if file_io.file_exists(self.url):
            pass  # TODO load dataset
        else:
            self.__create_dataset()

    def __create_dataset(self):
        pass
