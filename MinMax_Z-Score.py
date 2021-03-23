import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

print("SHREYAANS NAHATA: 19BCE2686\n")

names = ['height', 'weight']
dataset = pandas.read_csv(r'weight-height.csv', header=0, names=names)
array = dataset.values

print("------ Printing 10 headers of the dataset ------")
print(dataset.head(10))

print("\n------ Sumamrizing the dataset ------")
print(dataset.info())

# Min-Max Normalization
print("\n---- Min-Max Normalization ----")
X = array[:, 0:2]
Y = array[:, 1]
scaler = MinMaxScaler(feature_range=(-10,10))
rescaledX = scaler.fit_transform(X)

# summarize transformed data
print("Summarizing Transformed Data (10 headers): ")
np.set_printoptions(precision=3)
print(rescaledX[0:10, :])

# Z-Score Transformation
print("\n---- Z-Score Transformation ----")
X = array[:,0:2] 
Y = array[:,1] 
scaler = StandardScaler().fit(X) 
rescaledX = scaler.transform(X) 

# summarize transformed data
print("Summarizing Transformed Data (10 headers): ")
np.set_printoptions(precision=3) 
print(rescaledX[0:10,:])