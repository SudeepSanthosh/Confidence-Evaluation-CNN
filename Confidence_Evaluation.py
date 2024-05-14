from keras.models import Sequential
import tensorflow as tf

import tensorflow_datasets as tfds

# tf.enable_eager_execution()

from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
np.bool = np.bool_
import os
import cv2
import random 
from numpy import *
from PIL import Image
import theano

# path_test = os.chdir('C:/Users/91973/OneDrive/Desktop/Sudeep College Items/S8/FYP/archive.zip/sorted_data/sorted_data')
path_test = r'C:\Users\91973\OneDrive\Desktop\Sudeep College Items\S8\FYP\sorted_data\sorted_data'



Categories = ['confident','unconfident']
IMG_SIZE =200
# print(image_array.shape)

training = []
def createTrainingData():
  for category in Categories:
    path = os.path.join(path_test, category)
    class_num = Categories.index(category)
    for img in os.listdir(path):
      img_array = cv2.imread(os.path.join(path,img))
      # grayscale_img_array = cv2.cvtColor()
      new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
      training.append([new_array, class_num])
createTrainingData()

random.shuffle(training)

X =[]
y =[]
for features, label in training:
  X.append(features)
  y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

X = X.astype('float32')
X /= 255
Y = to_categorical(y, 4)
# print(Y[100])
# print(shape(Y))
X_np = np.array(X)
y_np = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size = 0.2, random_state = 4)
print("Dataset prepared successfully")
batch_size = 100
# nb_classes = 4
nb_epochs = 1
img_rows, img_columns = 200, 200
img_channel = 3
nb_filters = 32
nb_pool = 2
nb_conv = 3

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
#                            input_shape=(200, 200, 3)),
#     tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#     tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
#     tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(2,  activation=tf.nn.softmax)
# ])

model = Sequential()

# 1 - Convolution layer
model.add(tf.keras.layers.Conv2D(64,(3,3), padding='same', input_shape=(200, 200, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(tf.keras.layers.Conv2D(128,(5,5), padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(tf.keras.layers.Conv2D(512,(3,3), padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(tf.keras.layers.Conv2D(512,(3,3), padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(tf.keras.layers.BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(tf.keras.layers.BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = batch_size, epochs = nb_epochs, verbose = 1, validation_data = (X_test, y_test))

score = model.evaluate(X_test, y_test, verbose = 0 )
# print("Test Score: ", score[0])
# print("Test accuracy: ", score[1])
