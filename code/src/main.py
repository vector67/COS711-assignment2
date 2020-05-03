import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.pre_process import do_pre_process

tf.random.set_seed(1234)

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
    tf.keras.layers.Dense(9, activation='relu'),
    tf.keras.layers.Dense(9, activation='relu'),
    tf.keras.layers.Dense(9, activation='relu'),
    tf.keras.layers.Dense(9, activation='relu'),
    tf.keras.layers.Dense(9, activation='relu'),
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
model.fit(train_data, train_labels, batch_size=1020, epochs=500, validation_data=(test_data, test_labels))
print('evaluating: ')
print(model.evaluate(test_data, test_labels, verbose=2))
print(test_data[0], test_labels[0])
# for label in test_labels:
#     print(label)
for i in range(len(test_labels)):
    print('correct answer was', np.argmax(test_labels[i]), 'model predicted:', np.argmax(model.predict([[test_data[i]]])))
# print(model.layers[0].get_weights()[0])
# print(model.layers[1].get_weights()[0])
