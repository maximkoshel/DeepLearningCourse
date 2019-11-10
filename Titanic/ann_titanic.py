# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Loading the data
dataset = pd.read_csv('train.csv')
dataset_train=pd.read_csv('test.csv')

dataset['Embarked'].replace('S','0',inplace=True)#Replace chars to ints
dataset['Embarked'].replace('C','1',inplace=True )
dataset['Embarked'].replace('Q','2', inplace=True)

dataset_train['Embarked'].replace('S','0',inplace=True)#Replace chars to ints
dataset_train['Embarked'].replace('C','1',inplace=True )
dataset_train['Embarked'].replace('Q','2', inplace=True)



#Taking care of missing data
dataset['Age'].fillna((dataset['Age'].mean()), inplace=True)
dataset['Age']=dataset['Age'].astype(int)
dataset['Embarked'].fillna(method='backfill', inplace=True)


#Taking care of missing data
dataset_train['Age'].fillna((dataset_train['Age'].mean()), inplace=True)
dataset_train['Age']=dataset_train['Age'].astype(int)
dataset_train['Embarked'].fillna(method='backfill', inplace=True)

enclude = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'] 
X = dataset.loc[:,enclude].values #Loading only Relevent information
y = dataset.iloc[:,1:2].values #Survived or not data

X_test= dataset_train.loc[:,enclude].values #Loading only Relevent information



#Encoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_Gender = LabelEncoder()
X[:, 1] = labelencoder_X_Gender.fit_transform(X[:, 1])#Encode Gender 

onehotencoder = OneHotEncoder(categorical_features = [6])#Splitting Embarked
X = onehotencoder.fit_transform(X).toarray()
X =X[:,1:]#Removing dummy veriable

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Evaluating
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #Cancel some nuerouns to minimize overfitting
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
paramaters = {'batch_size':[32],
              'epochs':[500],
              'optimizer':['adam']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid=paramaters,scoring = 'accuracy',
                           cv=10,)
grid_search = grid_search.fit(X,y)  
best_paramater = grid_search.best_params_ 
best_accuracy = grid_search.best_score_  
# (4 Neurons all hidden layers)
#First evluation - 
#best paramaters : batch_size:32,epochs:500,optimizor:adam, best accuracy 0.820
#2 Hidden layer
#Second evaluation - 
#add one additional layer + 0.1 drop to prevent overfittng 
#best paramaters : batch_size:32,epochs:500,optimizor:adam, best accuracy 0.797
#3 Hidden layers
#Third evaluation - 
#2 Hidden layers
#best paramaters : batch_size:10,epochs:750,optimizor:adam, best accuracy 0.813
#Fourth evaluation - 
#3 Hidden layers
#Best paramaters : batch_size:64,epochs:500,optimizor:adam, best accuracy 0.795
#Fifth evaluation - 
#1 Hidden layer
#Best parameters : batch_size:32,epochs:500,optimizor:adam, best accuracy 0.815
#Sixth evaluation - 
#3 Hidden layer
#Best parameters : batch_size:15,epochs:500,optimizor:adam, best accuracy 0.756
#7 - 
#2 Hidden layers
#Best paramaters : batch_size:30,epochs:500,optimizor:adam, best accuracy 0.820
#8 - 
#2 Hidden layers,2 neurons, accuracy : 0.722 , batch : 30 , epochs : 500
#9 - 
#2 Hidden layers,5 neurons, accuracy : 0.817 , batch : 30 , epochs : 500
#10 - 
#2 Hidden layers,8 neurons, accuracy : 0.815 , batch : 30 , epochs : 500
#.............
#14
#1 hidden layer, 4 nuerons , batch:32,accuracy :0.830

classifier = Sequential()
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

