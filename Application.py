import cv2
import time
import tensorflow as tf
import numpy as np
import sys

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = tf.keras.models.load_model('pickle.model')


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Split = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(60, 60))

    Face = []
    for (x, y, w, h) in Split:
        Face.append(gray[y:y+h, x:x+w])
        for i in Face:
            i = cv2.resize(i, (50, 50))
            Instance = np.asarray(i)
            Instance = np.reshape(Instance, (1, 50, 50, 1))
            Instance = Instance / 255
            prediction = model.predict(Instance.astype('float32'))
            if prediction > 0.99:
                print('Welcome Adi')
            print(prediction)
            cv2.imshow('image', i)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
