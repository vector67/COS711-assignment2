import csv
import math
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

# tf.random.set_seed(1234)
plt.close('all')
path_to_data = '../resources/winequality_data/'
red_file_name = path_to_data + 'winequality-red.csv'
white_file_name = path_to_data + 'winequality-white.csv'

data = pd.read_csv(red_file_name, delimiter=';')

train_data, train_labels, test_data, test_labels = do_pre_process(data)
valid_layer_names = ["Dense"]
valid_layer_activation_functions = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu",
                                    "exponential"]


class Layer:
    def __init__(self, name, number_of_neurons, activation_function):
        self.name = name
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function

    def create_layer(self):
        if self.name == "Dense":
            return tf.keras.layers.Dense(self.number_of_neurons, activation=self.activation_function)
        elif self.name == "Conv1D":
            return tf.keras.layers.Conv1D(self.number_of_neurons, int(max(1, self.number_of_neurons / 10)))
        elif self.name == "MaxPooling1D":
            return tf.keras.layers.MaxPooling1D()
        elif self.name == "Dropout":
            return tf.keras.layers.Dropout(min(0.5, self.number_of_neurons / 100))
        else:
            raise Exception("nope, wrong layer type")

    def copy(self):
        return Layer(self.name, self.number_of_neurons, self.activation_function)

    def __repr__(self):
        return self.name + "(" + str(self.number_of_neurons) + ") " + self.activation_function


def create_sequential_layers(layer_def):
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(9, activation='tanh')])
    # model.add(, input_shape=(9,)))
    for layer in layer_def:
        model.add(layer.create_layer())

    model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['categorical_accuracy'])
    return model


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


def do_training(label, curriculum_data, curriculum_labels, num_curriculum_buckets, min_epochs, layers=None,
                batch_size=64):
    max_bucket_num = num_curriculum_buckets - 1
    print(layers)
    if layers is not None:
        model = create_sequential_layers(layers)
    else:
        model = create_compiled_model(label)

    start_time = time.time()
    past_val_loss = []
    epoch_batch_number = 0
    my_loop_batch_size = 5
    histories = []
    while True:
        epoch_number = epoch_batch_number * my_loop_batch_size
        # epoch_progress = epoch_number / min_epochs
        # current_curriculum_bucket = min(max_bucket_num, int(epoch_progress * num_curriculum_buckets))
        current_curriculum_data = curriculum_data
        current_curriculum_labels = curriculum_labels

        history = model.fit(current_curriculum_data, current_curriculum_labels, batch_size=batch_size,
                            initial_epoch=epoch_number,
                            epochs=epoch_number + my_loop_batch_size, validation_data=(test_data, test_labels),
                            verbose=0)
        print(len(current_curriculum_data), history.history['val_categorical_accuracy'][-1],
              history.history['val_loss'][-1])
        if math.isnan(history.history['val_loss'][-1]):
            final_accuracy = 0
            final_loss = 9999
            break
        past_val_loss.insert(0, history.history['val_loss'][-1])
        past_val_loss = past_val_loss[:20]
        comparison_range = 5
        if len(past_val_loss) > comparison_range:
            num_increases = 0
            for k in range(1, comparison_range + 1):
                if past_val_loss[k] - past_val_loss[k - 1] <= 0.0001:
                    num_increases += 1
            if num_increases / comparison_range >= 0.6:
                if epoch_number < min_epochs:
                    print('would have broken, but we aren\'t because min epochs hasn\'t been reached',
                          min_epochs - epoch_number, 'left')
                else:
                    print(past_val_loss)
                    print('ended with acc', history.history['val_categorical_accuracy'][-1])
                    final_accuracy = history.history['val_categorical_accuracy'][-1]
                    final_loss = history.history['val_loss'][-1]
                    break
        histories.append(history)
        epoch_batch_number += 1
    print('time', time.time() - start_time)
    return final_loss, final_accuracy, histories
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


bss = [8, 16, 32, 64, 128, 256, 512, 1024]
# bss = [64]
for i in range(0, 30):
    for bs in bss:
        start_time = time.time()
        loss, accuracy, histories = do_training('tanh', train_data, train_labels, 5, 50, batch_size=bs)
        total_time = time.time() - start_time
        losses = []
        accuracies = []
        for history in histories:
            # print(history.history)
            losses.extend(history.history['val_loss'])
            accuracies.extend(history.history['val_categorical_accuracy'])

        # print(losses)
        # print(accuracies)
        filename = "../output/batch_size_" + str(bs) + "_loss_curves.csv"
        if not os.path.exists(filename):
            open(filename, 'w').close()
        with open(filename, newline='\n') as csvfile:
            reader_rows = csv.reader(csvfile, delimiter=',', quotechar='"')
            rows = []
            for row in reader_rows:
                rows.append(row)
        rows.append(["|".join(map(str, losses)), "|".join(map(str, accuracies)), total_time])
        print(rows)
        # writer = csv.writer(open(filename, "wb"))
        # writer.writerows(np.array(loss_accuracy_curve))
        np.savetxt(filename, rows, delimiter=',', fmt='%s')

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
#
# def score_population(population_to_be_scored):
#     scores = [score for i, score in population_to_be_scored]
#     counter = 0
#     for individual in population_to_be_scored:
#         individual_genes = individual[0]
#         if individual[1] == 0:
#             # fitness calculation:
#             print(train_data.shape)
#             print(train_labels.shape)
#             # scores[counter] = do_training('tanh', train_data, train_labels, num_buckets, 200, layers=individual[0])
#             scores[counter] = random.randint(0, 100)
#         counter += 1
#     return scores
#
#
# def sort_second(val):
#     return val[1]
#
#
# chance_for_name_mutation = 0.2
# chance_for_activation_function_mutation = 0.2
# chance_to_gain_layer = 0.02
# chance_to_lose_layer = 0.02
#
#
# def random_layer():
#     return Layer(random.choice(valid_layer_names),
#                  random.randint(0, 100),
#                  random.choice(valid_layer_activation_functions))
#
#
# def mutate_layers(individual):
#     for ind_layer in individual:
#         if random.random() < mutation_rate:
#             attribute_to_change = "num_neurons"
#             if random.random() < chance_for_name_mutation:
#                 attribute_to_change = "name"
#             elif random.random() < chance_for_activation_function_mutation / chance_for_name_mutation:
#                 attribute_to_change = "activation"
#             if attribute_to_change == "num_neurons":
#                 ind_layer.number_of_neurons += int(math.pow(random.randint(0, 5), 1.5) - 5.6)
#             elif attribute_to_change == "name":
#                 ind_layer.name = random.choice(valid_layer_names)
#             elif attribute_to_change == "activation":
#                 ind_layer.activation_function = random.choice(valid_layer_activation_functions)
#     if random.random() < chance_to_gain_layer:
#         individual.append(random_layer())
#     if random.random() < chance_to_lose_layer:
#         selection = random.randint(0, len(individual)-1)
#         individual.pop(selection)
#
#
# def crossover_layers(individual1, individual2):
#     new_individual1 = individual1[: int(len(individual1) / 2)] + individual2[int(len(individual2) / 2):]
#     new_individual2 = individual2[: int(len(individual2) / 2)] + individual1[int(len(individual1) / 2):]
#     return new_individual1, new_individual2
#
#
# # do_training('tanh')
# num_training_patterns = len(train_data)
# num_buckets = 5
#
# curriculum = np.arange(num_training_patterns)
#
# population_size = 4
# population = []
# num_generations = 1
# mutation_rate = 0.1
# mutation_vs_crossover_rate = 0.5
#
# num_threads = 4
#
# filename = '../output/best_architecture.csv'
# if os.path.exists(filename):
#     with open(filename, 'r') as f:
#         for line in f:
#             line_arr = line.split(',')
#             population.append([list(map(int, line_arr[0].split("|"))), float(line_arr[1])])
#         print(population)
#
# for i in range(population_size - len(population)):
#     genes = []
#     for k in range(random.randint(0,10)):
#         genes.append(random_layer())
#     population.append([genes, 0])
#
# if __name__ == '__main__':
#     for generation_number in range(num_generations):
#         print('\n\n\nStarting generation:', generation_number)
#         population = population[:population_size]
#         print(population)
#         with Pool(num_threads) as p:
#             new_scores = np.array(p.map(score_population, np.split(np.array(population), num_threads))).flatten()
#         population = [[ind[0], score] for [ind, score] in zip(population, new_scores)]
#         population.sort(key=sort_second, reverse=True)
#         population = population[:int(population_size / 2)]
#         for k in range(population_size - len(population)):
#             if random.random() < mutation_vs_crossover_rate:
#                 newbie_genes1, newbie_genes2 = crossover_layers(population[np.random.choice(len(population))][0],
#                                                                 population[np.random.choice(len(population))][0])
#                 population.append([newbie_genes1, 0])
#                 k += 1
#                 if k < population_size - len(population):
#                     population.append([newbie_genes2, 0])
#             else:
#                 parent = population[np.random.choice(len(population))]
#                 newbie_genes = []
#                 for layer in parent[0]:
#                     newbie_genes.append(layer.copy())
#                 mutate_layers(newbie_genes)
#                 # population.append([newbie_genes, parent[1] + random.randint(-10, 10)])
#                 population.append([newbie_genes, 0])
#     print(population)
# print(population[0])
# final_array = []
# for pop in population:
#     final_array.append(["|".join(map(str, pop[0])), pop[1]])
# np.savetxt('../output/best_layers.csv', final_array, delimiter=',', fmt='%s')


# num_training_patterns = len(train_data)
# num_buckets = 5
#
# curriculum = np.arange(num_training_patterns)
#
# population_size = 40
# population = []
# num_generations = 30
# mutation_rate = 0.1
#
# num_threads = 4
#
# filename = '../output/best_order.csv'
# if os.path.exists(filename):
#     with open(filename, 'r') as f:
#         for line in f:
#             line_arr = line.split(',')
#             population.append([list(map(int, line_arr[0].split("|"))), float(line_arr[1])])
#         print(population)
#
# for i in range(population_size - len(population)):
#     genes = np.arange(num_training_patterns)
#     np.random.shuffle(genes)
#     population.append([genes, 0])
#
# if __name__ == '__main__':
#     for generation_number in range(num_generations):
#         print('\n\n\nStarting generation:', generation_number)
#
#         with Pool(num_threads) as p:
#             new_scores = np.array(p.map(score_population, np.split(np.array(population), num_threads))).flatten()
#         population = [[ind[0], score] for [ind, score] in zip(population, new_scores)]
#         population.sort(key=sort_second, reverse=True)
#         population = population[:int(population_size / 2)]
#         for k in range(population_size - len(population)):
#             parent = population[np.random.choice(len(population))]
#             newbie_genes = np.copy(parent[0])
#             mutate(newbie_genes, num_training_patterns)
#             # population.append([newbie_genes, parent[1] + random.randint(-10, 10)])
#             population.append([newbie_genes, 0])
#     print(population)
# print(population[0])
# final_array = []
# for pop in population:
#     final_array.append(["|".join(map(str, pop[0])), pop[1]])
# np.savetxt('../output/best_order.csv', final_array, delimiter=',', fmt='%s')
