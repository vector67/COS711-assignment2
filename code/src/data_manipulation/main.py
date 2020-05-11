import csv
import matplotlib.pyplot as plt
import numpy as np
bss = [8, 16, 32, 64, 128, 256, 512, 1024]
legend_loss = []
legend_accuracy = []
val_times = []
for bs in bss:
    val_loss_arr = []
    val_categorical_accuracy_arr = []
    with open('../../output/batch_size_' + str(bs) + '_loss_curves.csv', 'r') as f:
        data = list(csv.reader(f))
        counter = 0
        val_times_total = 0
        for row in data:
            val_losses = list(map(float, row[0].split("|")))
            val_categorical_accuracies = list(map(float, row[1].split("|")))
            val_times_total += float(row[2])
            for i in range(len(val_losses)):
                if len(val_loss_arr) <= i:
                    val_loss_arr.extend([[] for k in range(i - len(val_loss_arr)+1)])
                    val_categorical_accuracy_arr.extend([[] for k in range(i - len(val_categorical_accuracy_arr)+1)])
                # print(val_loss_arr, i)
                val_loss_arr[i].append(val_losses[i])
                val_categorical_accuracy_arr[i].append(val_categorical_accuracies[i])
            counter += 1
        previous_len = 0
        for i in range(len(val_loss_arr)):
            if len(val_loss_arr[i]) < 8:
                val_loss_arr = val_loss_arr[:i]
                val_categorical_accuracy_arr = val_categorical_accuracy_arr[:i]
                break
            previous_len = len(val_loss_arr[i])
            val_loss_arr[i] = sum(val_loss_arr[i]) / len(val_loss_arr[i])
            val_categorical_accuracy_arr[i] = sum(val_categorical_accuracy_arr[i]) / \
                                              len(val_categorical_accuracy_arr[i])
        # print(val_loss_arr)
        # print(val_categorical_accuracy_arr)

    # legend.append('loss_' + label + '(' + str(int(total)) + ')')
    # plt.plot(val_loss_arr)
    legend_loss.append('loss_' + str(bs) + '(' + str(int(counter)) + ')')
    legend_accuracy.append('accuracy_' + str(bs) + '(' + str(int(counter)) + ')')
    plt.figure(2)
    plt.plot(val_categorical_accuracy_arr)
    plt.figure(1)
    plt.plot(val_loss_arr)
    val_times.append(val_times_total / counter)
    counter += 1

plt.figure(1)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(legend_loss, loc='upper right')
plt.savefig('../../output/batch_size_loss.png')

plt.figure(2)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(legend_accuracy, loc='lower right')
plt.savefig('../../output/batch_size_accuracy.png')

# plt.legend(legend_accuracy, loc='lower right')
# plt.show()

people = bss
y_pos = np.arange(len(val_times))
error = np.std(val_times)

plt.figure(3)
fig, ax = plt.subplots()
ax.bar(y_pos, val_times, yerr=error, align='center')
ax.set_xticks(y_pos)
ax.set_xticklabels(people)
# ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Batch size')
ax.set_ylabel('Time to run (s)')
ax.set_title('Model run times in seconds')

plt.tight_layout()
plt.savefig('../../output/batch_size_times.png')
