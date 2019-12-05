from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()

#Convlolution
classifier.add(Convolution2D(filters=32,kernel_size=[3,3],input_shape=(64,64,3),activation='relu'))

#Adding second conv layer
classifier.add(Convolution2D(filters=32,kernel_size=[3,3],activation='relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#flattening
classifier.add((Flatten()))

#Full connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer='adam',loss= 'binary_crossentropy',metrics=['accuracy'])

#Fitting the CNN to the images 

from keras.preprocessing.image import ImageDataGenerator #Image augmantation,confuses the system
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:\dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:\dataset/training_Set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator( 
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)


#Making single prediction 
import numpy as np 
from keras.preprocessing import image 
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64, 64))
test_image = image.img_to_array(test_image)#added the 3rd dimansion, line 11 (64,64,3)
test_image = np.expand_dims(test_image,axis = 0)
result = classifier.predict(test_image)
training_set.class_indices #show what the paramters 
if result [0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'























