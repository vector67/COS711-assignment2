import csv
import matplotlib.pyplot as plt

labels = ['relu', 'tanh', 'sigmoid', 'linear']
legend = []
for label in labels:
    val_loss_arr = []
    val_categorical_accuracy_arr = []
    with open('../../output/loss_curves_' + label + '_hidden.csv', 'r') as f:
        data = list(csv.reader(f))
        counter = 0
        for row in data:
            val_loss_sum = 0
            val_categorical_accuracy_sum = 0
            if not counter == 0:
                for i in range(int(len(row)/2)):
                    val_loss_sum += float(row[i*2])
                    val_categorical_accuracy_sum += float(row[i*2 + 1])
                total = len(row)/2
                val_loss_arr.append(val_loss_sum / total)
                val_categorical_accuracy_arr.append(val_categorical_accuracy_sum / total)
            counter += 1
        print(val_loss_arr)
        print(val_categorical_accuracy_arr)

    # legend.append('loss_' + label + '(' + str(int(total)) + ')')
    # plt.plot(val_loss_arr)
    legend.append('accuracy_' + label + '(' + str(int(total)) + ')')
    plt.plot(val_categorical_accuracy_arr)
    counter += 1
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(legend, loc='upper left')
# plt.show()
plt.savefig('../../output/accuracy.png')