import csv
import os
import random
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from multiprocessing import Pool

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


def create_compiled_model(label):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(9, activation='tanh'),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(9, activation=label),
        tf.keras.layers.Dense(10, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['categorical_accuracy'])
    return model


def do_training(label, curriculum_data, curriculum_labels, num_curriculum_buckets, min_epochs):
    max_bucket_num = num_curriculum_buckets - 1

    model = create_compiled_model(label)

    start_time = time.time()
    past_val_loss = []
    epoch_batch_number = 0
    my_loop_batch_size = 5

    while True:
        epoch_number = epoch_batch_number * my_loop_batch_size
        epoch_progress = epoch_number / min_epochs
        current_curriculum_bucket = min(max_bucket_num, int(epoch_progress * num_curriculum_buckets))
        current_curriculum_data = curriculum_data[current_curriculum_bucket]
        current_curriculum_labels = curriculum_labels[current_curriculum_bucket]

        history = model.fit(current_curriculum_data, current_curriculum_labels, batch_size=batch_size,
                            initial_epoch=epoch_number,
                            epochs=epoch_number + my_loop_batch_size, validation_data=(test_data, test_labels),
                            verbose=0)
        print(len(current_curriculum_data), history.history['val_categorical_accuracy'][-1])
        past_val_loss.insert(0, history.history['val_loss'][-1])
        past_val_loss = past_val_loss[:20]
        comparison_range = 5
        if len(past_val_loss) > comparison_range:
            num_increases = 0
            for k in range(1, comparison_range + 1):
                if past_val_loss[k] - past_val_loss[k - 1] < 0:
                    num_increases += 1
            if num_increases / comparison_range >= 0.6:
                if epoch_number < min_epochs:
                    print('would have broken, but we aren\'t because min epochs hasn\'t been reached')
                else:
                    print(past_val_loss)
                    print('ended with acc', history.history['val_categorical_accuracy'][-1])
                    final_accuracy = history.history['val_categorical_accuracy'][-1]
                    break
        epoch_batch_number += 1
    print('time', time.time() - start_time)
    return final_accuracy
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

def score_population(population_to_be_scored):
    scores = [score for i, score in population_to_be_scored]
    counter = 0
    for individual in population_to_be_scored:
        individual_genes = individual[0]
        if individual[1] == 0:
            train_data_genes = np.split(train_data[individual_genes], num_buckets)
            train_labels_genes = np.split(train_labels[individual_genes], num_buckets)
            for b in range(1, num_buckets):
                train_data_genes[b] = np.concatenate((train_data_genes[b - 1], train_data_genes[b]))
                train_labels_genes[b] = np.concatenate((train_labels_genes[b - 1], train_labels_genes[b]))
            # fitness calculation:
            scores[counter] = do_training('tanh', train_data_genes, train_labels_genes, num_buckets, 200)
            # scores[counter] = random.randint(0, 100)
        counter += 1
    return scores


def sort_second(val):
    return val[1]


def mutate(individual, num_genes):
    idx = np.random.choice(num_genes, int(num_genes * mutation_rate), replace=False)
    individual[idx] = np.random.permutation(individual[idx])


# do_training('tanh')
num_training_patterns = len(train_data)
num_buckets = 5

curriculum = np.arange(num_training_patterns)

population_size = 40
population = []
num_generations = 30
mutation_rate = 0.1

num_threads = 4

filename = '../output/best_order.csv'
if os.path.exists(filename):
    with open(filename, 'r') as f:
        for line in f:
            line_arr = line.split(',')
            population.append([list(map(int, line_arr[0].split("|"))), float(line_arr[1])])
        print(population)

for i in range(population_size - len(population)):
    genes = np.arange(num_training_patterns)
    np.random.shuffle(genes)
    population.append([genes, 0])

if __name__ == '__main__':
    for generation_number in range(num_generations):
        print('\n\n\nStarting generation:', generation_number)

        with Pool(num_threads) as p:
            new_scores = np.array(p.map(score_population, np.split(np.array(population), num_threads))).flatten()
        population = [[ind[0], score] for [ind, score] in zip(population, new_scores)]
        population.sort(key=sort_second, reverse=True)
        population = population[:int(population_size / 2)]
        for k in range(population_size - len(population)):
            parent = population[np.random.choice(len(population))]
            newbie_genes = np.copy(parent[0])
            mutate(newbie_genes, num_training_patterns)
            # population.append([newbie_genes, parent[1] + random.randint(-10, 10)])
            population.append([newbie_genes, 0])
    print(population)
print(population[0])
final_array = []
for pop in population:
    final_array.append(["|".join(map(str, pop[0])), pop[1]])
np.savetxt('../output/best_order.csv', final_array, delimiter=',', fmt='%s')

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
