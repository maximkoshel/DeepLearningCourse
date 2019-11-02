# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(6,input_shape = (11,),kernel_initializer='uniform',activation = 'relu'))#First layer, 11 input layers and 6 output layers
#Adding the second layer
classifier.add(Dense(6,kernel_initializer='uniform',activation = 'relu'))#Second layer, dont need input layers because know from previous
#Adding the output layer
classifier.add(Dense(1,kernel_initializer='uniform',activation = 'sigmoid'))#Last layer, 1 output
#compiling the ANN
classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

#fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Predicting one person
"France,credit score =600,Male,40,3,60000,2,yes,yes,es salary:5000"

new_prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,6000,2,1,1,5000]])))
new_prediction= (new_prediction>0.5)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
