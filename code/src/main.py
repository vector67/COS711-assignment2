import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from src.constants import PERCENTAGE_TRAIN
from src.pre_process import do_pre_process

plt.close('all')

path_to_data = '../resources/winequality_data/'
red_file_name = path_to_data + 'winequality-red.csv'
white_file_name = path_to_data + 'winequality-white.csv'

data = pd.read_csv(red_file_name, delimiter=';')

do_pre_process(data)
# train_data, train_labels, test_data, test_labels = do_pre_process(np.array_split(data.values, [int(len(data) * PERCENTAGE_TRAIN)]))
# label_train, label_test = data_train[:, -1], data_test[:, -1]
# data_train, data_test = data_train[:, :-1], data_test[:, :-1]
#
# label_train /= 10
# label_test /= 10
#
# min_max_scaler = preprocessing.MinMaxScaler()
# data_train_scaled = min_max_scaler.fit_transform(data_train)
# data_test_scaled = min_max_scaler.fit_transform(data_test)
# # data_train_scaled = data_train
# # data_test_scaled = data_test
# model = tf.keras.models.Sequential([
#   # tf.keras.layers.Dense(12, activation='sigmoid'),
#   # tf.keras.layers.Dense(1, activation='sigmoid'),
#   # tf.keras.layers.Dense(0, activation='sigmoid'),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
# loss_fn = tf.keras.losses.MeanSquaredError()
# print(data_train_scaled.shape)
# print(label_train.shape)
# print(label_train)
# model.compile(optimizer='adam',
#               loss=loss_fn,
#               metrics=['MeanSquaredError'])
# model.fit(data_train_scaled, label_train, epochs=500)
# print(model.evaluate(data_test_scaled, label_test, verbose=2))
#
# print(model.layers[0].get_weights()[0])
# # print(model.layers[1].get_weights()[0])
