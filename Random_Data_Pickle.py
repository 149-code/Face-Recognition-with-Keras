import cv2
import os
import sys
import pickle

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
N = 0
Path = 'Images/Random'

List = []
for img in os.listdir(Path):
    Current = cv2.imread(os.path.join(Path, img), cv2.IMREAD_GRAYSCALE)
    Split = faceCascade.detectMultiScale(Current,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30, 30))

    for (x, y, w, h) in Split:
        Face = Current[y:y+h, x:x+w]
        List.append([cv2.resize(Face, (50, 50)), 0])
    print(N); N += 1

pickle.dump(List, open('Random_Pickle.p', 'wb'))
