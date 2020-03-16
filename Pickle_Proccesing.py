import pickle
import cv2
import random
import numpy as np

#Put all the names of class pickles in Pickles_To_Process
Pickles_To_Process = ["Class1_Pickle.p", "Random_Pickle.p"]
Images = list()
for Pickle in Pickles_To_Process:
    Images.append(pickle.load(open(Pickle, 'rb')))

X = []
Y = []

random.shuffle(Images)
for data, lable in Images:
    X.append(data)
    Y.append(lable)

X = np.asarray(X).reshape(len(X), 50, 50, 1)
Y = np.asarray(Y)
X = X / 255

pickle.dump(X, open('X.p', 'wb'))
pickle.dump(Y, open('Y.p', 'wb'))
