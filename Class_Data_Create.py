import cv2
import time

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

N = 0
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
                if len(Face) < 2:
                        for i in Face:
                                Resize = cv2.resize(i, (50, 50))
                                cv2.imwrite('Images/Class1/' + str(N) + '.jpg', Resize)
                                N += 1
                                print(N)
cap.release()
cv2.destroyAllWindows()
