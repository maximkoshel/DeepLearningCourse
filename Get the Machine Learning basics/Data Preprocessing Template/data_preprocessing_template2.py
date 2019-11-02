#Data pre proccessing 

#Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Taking care of misssing data
from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer = imputer.fit(x[:,1:3])
x[:, 1:3]= imputer.transform(x[:,1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0]) #Countries become number (france = 0 , spain = 2 ,germany = 1)
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray() #countries divided to separate columns

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) #Tranform purchased column to 0,1