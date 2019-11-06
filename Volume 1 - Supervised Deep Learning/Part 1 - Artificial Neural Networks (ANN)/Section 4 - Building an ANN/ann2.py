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
from keras.layers import Dropout #Cancel some nuerouns to minimize overfitting


#Initialising the ANN
classifier = Sequential()
#Adding the input layer and the first hidden layer
classifier.add(Dense(6,input_shape = (11,),kernel_initializer='uniform',activation = 'relu'))#First layer, 11 input layers and 6 output layers
classifier.add(Dropout(rate = 0.1))#Cancel some nuerouns to minimize overfitting
#Adding the second layer
classifier.add(Dense(6,kernel_initializer='uniform',activation = 'relu'))#Second layer, dont need input layers because know from previous
classifier.add(Dropout(rate =  0.1))
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


#Evaluating the ANN------------------------------------------------------------
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)

mean = accuracies.mean()#average 
varieance = accuracies.std()#varience of the average
#------------------------------------------------------------------------------




#Tunning the system for best preformence and then showint the best one --------
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
paramaters = {'batch_size':[25,32],
              'epochs':[100,500],
              'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid=paramaters,scoring = 'accuracy',
                           cv=10)
grid_search = grid_search.fit(X_train,y_train)  
best_paramater = grid_search.best_params_    #will show the best paramaters to input for best accuracy
best_accuracy = grid_search.best_score_ #Will show the best accuracy 
#------------------------------------------------------------------------------
    
    
    
    
    
    
    
    
    
    
    