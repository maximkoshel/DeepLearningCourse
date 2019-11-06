# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')
enclude = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'] 
X = dataset.loc[:,enclude].values #Loading only Relevent information
y = dataset.iloc[:,1:2].values #Survived or not data

