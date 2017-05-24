# -*- coding: utf-8 -*

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv')
# x will be all the columns minus the last one
X = dataset.iloc[:, :-1].values
# y will be column 3
y = dataset.iloc[:, 3].values
                
# taking care of missing data
## this is a preprocessing package that cleans the data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
### : all the columns, 1:3, columns 1 and 2
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])