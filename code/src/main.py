import csv
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from constants import NUM_EPOCHS
from pre_process import do_pre_process

# tf.random.set_seed(1234)
label = 'linear'
for i in range(39):
    plt.close('all')

    path_to_data = '../resources/winequality_data/'
    red_file_name = path_to_data + 'winequality-red.csv'
    white_file_name = path_to_data + 'winequality-white.csv'

    data = pd.read_csv(red_file_name, delimiter=';')

    train_data, train_labels, test_data, test_labels = do_pre_process(data)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(9, activation='tanh'),
        # tf.keras.layers.Dense(100, activation='tanh'),
        # tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(10, activation='sigmoid')
    ])


    def weighted_loss(y_true, y_pred):
        return -tf.math.reduce_sum(y_true * tf.math.log(
            tf.math.exp(y_pred) / tf.reshape(tf.math.reduce_sum(tf.math.exp(y_pred), axis=-1), (-1, 1))), axis=-1)


    loss_fn = tf.keras.losses.categorical_crossentropy
    # loss_fn = weighted_loss

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['categorical_accuracy'])
    # print(model.predict(np.array([test_data[0]])))
    history = model.fit(train_data, train_labels, batch_size=64, epochs=NUM_EPOCHS,
                        validation_data=(test_data, test_labels))
    print(history.history)
    loss_accuracy_curve = []
    for epoch in range(NUM_EPOCHS):
        line = ""
        loss_value = history.history['val_loss'][epoch]
        accuracy_value = history.history['val_categorical_accuracy'][epoch]
        for key in history.history:
            line += " " + str(history.history[key][epoch])
        print(line)
        loss_accuracy_curve.append([loss_value, accuracy_value])
    filename = '../output/loss_curves_' + label + '_hidden.csv'
    if not os.path.exists(filename):
        open(filename, 'w').close()
    with open(filename, 'r') as f:
        data = list(csv.reader(f))
        counter = 0
        for row in data:
            if not counter == 0:
                row.append(loss_accuracy_curve[counter - 1][0])
                row.append(loss_accuracy_curve[counter - 1][1])
            else:
                col_num = str(int(row[-1][-2:]) + 1).zfill(2)
                row.append('val_loss_' + col_num)
                row.append('val_categorical_accuracy_' + col_num)
            counter += 1
        if counter == 0:

            data.append(['val_loss_00', 'val_categorical_accuracy_00'])
            for loss_acc in loss_accuracy_curve:
                data.append(loss_acc)
    np.savetxt('../output/loss_curves_' + label + '_hidden.csv', data, delimiter=',', fmt='%s')

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
