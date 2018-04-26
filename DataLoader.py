import os
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class DataLoader:
    def __init__(self, batch_size, test_split=0.1, debug=False):
        self.debug = debug
        self.data_folder = "../data/images_data_crop/"
        self.label_folder = "../data/images_mask_crop/"
        self.file_list = os.listdir(self.data_folder)
        self.train_split = 1 - test_split
        self.total_samples = len(self.file_list)
        if self.debug:
            self.training_set_size = 4  # a small set for debugging
            # self.test_data = self.file_list[self.training_set_size:8]
            self.test_data = self.file_list[:self.training_set_size]
        else:
            self.training_set_size = int(self.train_split * self.total_samples)
            self.test_data = self.file_list[self.training_set_size:]
        self.train_data = self.file_list[:self.training_set_size]
        self.test_set_size = len(self.test_data)
        self.training_batch_counter = 0
        self.test_batch_counter = 0
        self.batch_size = batch_size
        sum_train = 0
        for file_name in self.train_data:
            data = plt.imread(self.data_folder + file_name)
            if len(data.shape) < 3:
                data = np.stack((data,) * 3, 2)
            data = data / 255
            sum_train += data
        self.mean_train = sum_train / float(self.training_set_size)
        assert self.batch_size == int(self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_training_set_size(self):
        return self.training_set_size

    def shuffle_training_set(self):
        self.training_batch_counter = 0
        if not self.debug:
            self.train_data = np.random.permutation(self.train_data)

    def get_next_training_batch(self):
        X, y = [], []
        for i in range(self.training_batch_counter * self.batch_size,
                       min(self.training_set_size, (self.training_batch_counter + 1) * self.batch_size)):
            data = plt.imread(self.data_folder + str(self.train_data[i]))
            if len(data.shape) < 3:
                data = np.stack((data,) * 3, 2)
            data = data / 255
            data -= self.mean_train
            X.append(data)
            mask = loadmat(self.label_folder + str(self.train_data[i][:-4]) + '_mask')['mask']
            label = np.array([1-mask, mask], dtype='uint8')
            y.append(label)
        self.training_batch_counter += 1
        return np.moveaxis(np.array(X), -1, 1), np.array(y)

    def get_test_data_file_names(self):
        return self.test_data

    def get_test_set_size(self):
        return self.test_set_size

    def reset_test_batch_counter(self):
        self.test_batch_counter = 0

    def get_next_test_batch(self):
        X, y = [], []
        for i in range(self.test_batch_counter * self.batch_size,
                       min(self.test_set_size, (self.test_batch_counter + 1) * self.batch_size)):
            data = plt.imread(self.data_folder + str(self.test_data[i]))
            if len(data.shape) < 3:
                data = np.stack((data,) * 3, 2)
            data = data / 255
            data -= self.mean_train  # this still uses mean_train!
            X.append(data)
            mask = loadmat(self.label_folder + str(self.test_data[i][:-4]) + '_mask')['mask']
            label = np.array([1-mask, mask], dtype='uint8')
            y.append(label)
        self.test_batch_counter += 1
        return np.moveaxis(np.array(X), -1, 1), np.array(y)
