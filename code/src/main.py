import csv
import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.constants import NUM_EPOCHS
from src.pre_process import do_pre_process

batch_size = 64
# tf.random.set_seed(1234)
plt.close('all')
path_to_data = '../resources/winequality_data/'
red_file_name = path_to_data + 'winequality-red.csv'
white_file_name = path_to_data + 'winequality-white.csv'

data = pd.read_csv(red_file_name, delimiter=';')

train_data, train_labels, test_data, test_labels = do_pre_process(data)


def do_training(label):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(9, activation='tanh'),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(10, activation='sigmoid')
    ])

    loss_fn = tf.keras.losses.categorical_crossentropy

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['categorical_accuracy'])
    # print(model.predict(np.array([test_data[0]])))
    print(model.metrics_names)
    start_time = time.time()
    past_val_loss = []
    i = 0
    while True:
        # idx = np.random.randint(len(train_data), size=batch_size * 5)
        # something = model.train_on_batch(train_data[idx, :], train_labels[idx, :])
        history = model.fit(train_data, train_labels, batch_size=batch_size, initial_epoch=i * 5,
                            epochs=(i + 1) * 5, validation_data=(test_data, test_labels), verbose=0)
        print(history.history)
        # print(history.history['val_loss'][0],  history.history['loss'][0], history.history['val_loss'][0] / history.history['loss'][0])
        past_val_loss.insert(0, history.history['val_loss'][-1])
        past_val_loss = past_val_loss[:20]
        val_loss_improvement = past_val_loss[-1] - past_val_loss[0]
        comparison_range = 5
        if len(past_val_loss) > comparison_range:
            num_increases = 0
            for k in range(1, comparison_range + 1):
                if past_val_loss[k] - past_val_loss[k - 1] < 0:
                    num_increases += 1
            if num_increases / comparison_range >= 0.6:
                print(past_val_loss)
                print('failed improvement test with', val_loss_improvement)
                print('ended with acc', history.history['val_categorical_accuracy'][-1])
                break
        i += 1
    print('time', time.time() - start_time)
    # print(history.history)
    # loss_accuracy_curve = []
    # for epoch in range(NUM_EPOCHS):
    #     line = ""
    #     loss_value = history.history['val_loss'][epoch]
    #     accuracy_value = history.history['val_categorical_accuracy'][epoch]
    #     for key in history.history:
    #         line += " " + str(history.history[key][epoch])
    #     print(line)
    #     loss_accuracy_curve.append([loss_value, accuracy_value])
    # filename = '../output/loss_curves_' + label + '_hidden.csv'
    # if not os.path.exists(filename):
    #     open(filename, 'w').close()
    # with open(filename, 'r') as f:
    #     data = list(csv.reader(f))
    #     counter = 0
    #     for row in data:
    #         if not counter == 0:
    #             row.append(loss_accuracy_curve[counter - 1][0])
    #             row.append(loss_accuracy_curve[counter - 1][1])
    #         else:
    #             col_num = str(int(row[-1][-2:]) + 1).zfill(2)
    #             row.append('val_loss_' + col_num)
    #             row.append('val_categorical_accuracy_' + col_num)
    #         counter += 1
    #     if counter == 0:
    #
    #         data.append(['val_loss_00', 'val_categorical_accuracy_00'])
    #         for loss_acc in loss_accuracy_curve:
    #             data.append(loss_acc)
    # np.savetxt('../output/loss_curves_' + label + '_hidden.csv', data, delimiter=',', fmt='%s')


for i in range(1):
    do_training('tanh')

# writer = csv.writer(open("../output/loss_curves_relu_hidden.csv", "wb"))
# writer.writerows(np.array(loss_accuracy_curve))
# print('evaluating: ')
# print(model.evaluate(test_data, test_labels, verbose=2))
# print(test_data[0], test_labels[0])
# # for label in test_labels:
# #     print(label)
# for i in range(len(test_labels)):
#     print('correct answer was', np.argmax(test_labels[i]), 'model predicted:',
#           np.argmax(model.predict([[test_data[i]]])))
# print(model.layers[0].get_weights()[0])
# print(model.layers[1].get_weights()[0])
