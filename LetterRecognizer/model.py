import tensorflow as tf
from Tensorflow_DataConverter.load.converter_lab import labels_to_binary


class Model:
    def load_data(self, train_data_set, test_data_set):
        train_data_set._labels = labels_to_binary(train_data_set.labels)
        test_data_set._labels = labels_to_binary(test_data_set.labels)
        self.train_set = train_data_set
        self.test_set = test_data_set

    def train(self):
        pass