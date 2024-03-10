from pandas import read_csv
from pandas import set_option
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names) #read data from pima_indians
array = data.values
peek = data.head(5)
print(peek) #print the first 5 samples

set_option('display.width',100)
correlations = data.corr(method='pearson')
print(correlations) #calculate correlation

X=array[:,0:8]
Y=array[:,8:]
scaler = MinMaxScaler(feature_range=(0,1))
scaledX = scaler.fit_transform(X)
set_printoptions(precision=3)
print(scaledX[0:5,:]) #rescale features & print the features of first 5 samples

import pandas as pd
rescaleDataframe = pd.DataFrame(scaledX)
correlation_scaled = rescaleDataframe.corr(method='pearson')
print(correlation_scaled)

#group data & calculate correlation of the feature in Class0
classgroup = data.groupby('class')
class_0 = classgroup.get_group(0)
print(class_0.head(10))

correlation_class0 = class_0.corr(method='pearson')
print(correlation_class0)

#group data & calculate correlation of the feature in Class1
classgroup = data.groupby('class')
class_1 = classgroup.get_group(1)
print(class_1.head(10))

correlation_class1 = class_1.corr(method='pearson')
print(correlation_class1)