import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os, pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X = pickle.load(open('X.p', 'rb'))
Y = pickle.load(open('Y.p', 'rb'))

Model = tf.keras.models.Sequential()

Model.add(Conv2D(256, (7, 7)))
Model.add(Activation('relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))

Model.add(Conv2D(256, (5, 5)))
Model.add(Activation('relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))

Model.add(Conv2D(256, (3, 3)))
Model.add(Activation('relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))

Model.add(Flatten())
Model.add(Dense(64))
Model.add(Dense(16))

Model.add(Dense(len(set(Y))))
Model.add(Activation('sigmoid'))
Model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

Model.fit(X, Y,
          epochs = 10,
          validation_split=0.1,
          batch_size = 100,
          verbose = 1)

Model.save('pickle2.model')
