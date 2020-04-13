import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from src.constants import PERCENTAGE_TRAIN


def normalize(data):
    # x = np.random.random_integers(1, 100, 5)
    for i in range(11):
        # normalized = data[:, i] / np.linalg.norm(data[:, i])
        normalized = preprocessing.normalize(data[:, i, None], norm='max', axis=0)
        print(normalized, np.min(normalized), np.max(normalized))
        plt.hist(normalized, bins=20)
        plt.ylabel('No of times')
        plt.show()
    # return normalized


def pre_process(data_all):
    data, labels = data_all[:, :-1], data_all[:, -1]
    final_data = normalize(data)
    return final_data, labels


def do_pre_process(data):
    train_data_all, test_data_all = np.array_split(data.values, [int(len(data) * PERCENTAGE_TRAIN)])
    print('len train', len(train_data_all), 'len test', len(test_data_all))
    train_data, train_labels = pre_process(train_data_all)
    # test_data, test_labels = pre_process(test_data_all)
    return train_data, train_labels  # , test_data, test_labels
