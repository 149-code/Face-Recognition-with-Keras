import cv2
import os
import sys
import pickle

N = 0
Path = 'Images/Class1'

List = []
for img in os.listdir(Path):
    try:
        Current = cv2.imread(os.path.join(Path, img), cv2.IMREAD_GRAYSCALE)
        print(img)
        Current = cv2.resize(Current, (50, 50))
        List.append([Current, 1])
        N += 1
    except:
        pass

print(len(List))
pickle.dump(List, open('Class_Pickle.p', 'wb'))
