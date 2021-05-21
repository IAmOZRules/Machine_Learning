from pandas import read_csv
import numpy as np
 
dataset = read_csv(r'pima-indians-diabetes.csv', header=None)

print("SHREYAANS NAHATA: 19BCE2686\n")

print("------ Printing 10 headers of the dataset ------")
print(dataset.head(10))

print("\n------ Summarizing the dataset ------")
print(dataset.info())

print("\n------ Counting the number of missing values in each column ------")
print((dataset[list(range(0,9))] == 0).sum())

print("\n------ Replacing 0 values with NaN ------")
for column in range(9):
    dataset[column].replace(0, np.nan, inplace=True)
print(dataset.head(5))

print("\n------ Replacing NaN values with Mean of respective column ------")
for column in range(9):
    mean_value = dataset[column].mean()
    dataset[column] = dataset[column].fillna(mean_value)
print(dataset.head(5))