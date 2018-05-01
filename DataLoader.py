import os
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEBUG_SET_SIZE = 4
TRAIN_MODE = 'train'
TEST_MODE = 'test'
STR_TRAIN_BATCH_COUNTER = 'train_batch_counter'
STR_TRAIN_SET_SIZE = 'train_set_size'
STR_TRAIN_DATA = 'train_data'
STR_TEST_BATCH_COUNTER = 'test_batch_counter'
STR_TEST_SET_SIZE = 'test_set_size'
STR_TEST_DATA = 'test_data'


class DataLoader:
    def __init__(self, batch_size, test_split=0.1, use_6_channels=True, debug=False):
        assert isinstance(batch_size, int)
        self.batch_size = batch_size
        self.debug = debug
        self.use_6_channels = use_6_channels
        self.test_split = test_split
        
        if self.use_6_channels:
            self.data_folder = "../data/6_channel_data/"  # pre-processed (zero-mean, normalized, position and shape channels)
        else:
            self.data_folder = "../data/images_data_crop/"
        self.images_folder = "../data/images_data_crop/"
        self.label_folder = "../data/images_mask_crop/"
        
        self.file_list = os.listdir(self.data_folder)
        self.total_samples = len(self.file_list)
        
        self.state_dict = self.get_state_dict()
        
        if not self.use_6_channels:
            sum_train = 0
            for file_name in self.state_dict[STR_TRAIN_DATA]:
                data = plt.imread(self.data_folder + file_name)
                if len(data.shape) < 3:
                    data = np.stack((data,) * 3, 2)
                data = data / 255
                sum_train += data
            self.mean_train = sum_train / float(self.state_dict[STR_TRAIN_SET_SIZE])
    
    def get_state_dict(self):
        state_dict = dict()

        if self.debug:
            state_dict[STR_TRAIN_SET_SIZE] = DEBUG_SET_SIZE  # a small set for debugging
            state_dict[STR_TEST_DATA] = self.file_list[:state_dict[STR_TRAIN_SET_SIZE]]  # same as training set
        else:
            state_dict[STR_TRAIN_SET_SIZE] = int((1 - self.test_split) * self.total_samples)
            state_dict[STR_TEST_DATA] = self.file_list[state_dict[STR_TRAIN_SET_SIZE]:]
        state_dict[STR_TRAIN_DATA] = self.file_list[:state_dict[STR_TRAIN_SET_SIZE]]
        state_dict[STR_TEST_SET_SIZE] = len(state_dict[STR_TEST_DATA])
        state_dict[STR_TRAIN_BATCH_COUNTER] = 0
        state_dict[STR_TEST_BATCH_COUNTER] = 0

        return state_dict

    def get_batch_size(self):
        return self.batch_size

    def get_training_set_size(self):
        return self.state_dict[STR_TRAIN_SET_SIZE]

    def shuffle_training_set(self):
        self.state_dict[STR_TRAIN_BATCH_COUNTER] = 0
        if not self.debug:
            np.random.shuffle(self.state_dict[STR_TRAIN_DATA])

    def get_next_training_batch(self):
        return self.get_next_batch(TRAIN_MODE)

    def get_test_data_file_names(self):
        return self.state_dict[STR_TEST_DATA]

    def get_test_set_size(self):
        return self.state_dict[STR_TEST_SET_SIZE]

    def reset_test_batch_counter(self):
        self.state_dict[STR_TEST_BATCH_COUNTER] = 0

    def get_next_test_batch(self):
        return self.get_next_batch(TEST_MODE)

    def get_next_batch(self, mode):
        if mode == TRAIN_MODE:
            str_batch_counter = STR_TRAIN_BATCH_COUNTER
            str_set_size = STR_TRAIN_SET_SIZE
            str_data = STR_TRAIN_DATA
        elif mode == TEST_MODE:
            str_batch_counter = STR_TEST_BATCH_COUNTER
            str_set_size = STR_TEST_SET_SIZE
            str_data = STR_TEST_DATA
        else:
            raise ValueError('Invalid dataset mode. Unable to fetch next batch.')
            
        X, y = [], []
        for i in range(self.state_dict[str_batch_counter] * self.batch_size,
                       min(self.state_dict[str_set_size], (self.state_dict[str_batch_counter] + 1) * self.batch_size)):
            if self.use_6_channels:
                data = loadmat(self.data_folder + str(self.state_dict[str_data][i]))
                data = data['img']
            else:
                data = plt.imread(self.data_folder + str(self.state_dict[str_data][i]))
                if len(data.shape) < 3:
                    data = np.stack((data,) * 3, 2)
                data = data / 255
                data -= self.mean_train
            X.append(data)
            mask = loadmat(self.label_folder + str(self.state_dict[str_data][i][:-4]) + '_mask')['mask']
            y.append(mask.astype('int64'))
        self.state_dict[str_batch_counter] += 1
        return np.moveaxis(np.array(X), -1, 1), np.array(y)