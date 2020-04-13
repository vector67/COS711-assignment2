import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')

path_to_data = '../resources/winequality_data/'
red_file_name = path_to_data + 'winequality-red.csv'
white_file_name = path_to_data + 'winequality-white.csv'

data = pd.read_csv(red_file_name, delimiter=';')
print(data)
print(data['volatile acidity'])
pd.DataFrame.hist(data, 'volatile acidity')