import numpy as np
from scipy import stats
from scipy.stats import zscore
from sklearn.decomposition import PCA

from constants import PERCENTAGE_TRAIN, PERCENTAGE_GLOBAL_TEST


def normalize(data):
    print('data', data)
    data = data[(np.abs(stats.zscore(data[:, :-1])) < 3).all(axis=1)]
    data_zscore = zscore(data[:, :-1], axis=0, ddof=1)
    for i in range(11):
        max = np.max(data_zscore[:, i])
        min = np.min(data_zscore[:, i])
        normalized = (data_zscore[:, i] - min) / (max - min) * 2 - 1
        data[:, i] = normalized
        # plt.hist(data[:, i], bins=20)
        # plt.ylabel('No of times')
        # plt.show()
    return data


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def pre_process(data_all):
    data_all = normalize(data_all)
    data, labels = data_all[:, :-1], data_all[:, -1]
    labels = get_one_hot(labels.astype(int), 10)
    return data, labels


# split normalization
# def do_pre_process(data):
#     train_data_all, test_data_all = np.array_split(data.values, [int(len(data) * PERCENTAGE_TRAIN)])
#     test_data_all, global_test_data_all = np.array_split(test_data_all,
#                                                          [int(len(test_data_all) * PERCENTAGE_GLOBAL_TEST)])
#     print('len train', len(train_data_all), 'len test', len(test_data_all))
#     train_data, train_labels = pre_process(train_data_all)
#     pca = PCA(.95)
#     pca.fit(train_data)
#     test_data, test_labels = pre_process(test_data_all)
#
#     train_data = pca.transform(train_data)
#     test_data = pca.transform(test_data)
#
#     print(len(train_data), len(train_labels), len(test_data), len(test_labels))
#     return train_data, train_labels, test_data, test_labels

# don't split normalization
def do_pre_process(data):
    # print('len train', len(train_data_all), 'len test', len(test_data_all))
    data_processed, labels = pre_process(np.array(data))
    pca = PCA(.95)
    pca.fit(data_processed)
    # data_processed = pca.transform(data_processed)

    train_data, test_data = np.array_split(data_processed, [int(len(data_processed) * PERCENTAGE_TRAIN)])
    test_data, global_test_data = np.array_split(test_data, [int(len(test_data) * PERCENTAGE_GLOBAL_TEST)])

    train_labels, test_labels = np.array_split(labels, [int(len(labels) * PERCENTAGE_TRAIN)])
    test_labels, global_labels = np.array_split(test_labels, [int(len(test_labels) * PERCENTAGE_GLOBAL_TEST)])

    print(len(train_data), len(train_labels), len(test_data), len(test_labels))
    return train_data, train_labels, test_data, test_labels
