# Face-Recognition-with-Keras

A folder of Python Programms to create and train a face recognition model


Includes:

Class_Data_Create.py #Uses a camera connected to your computer to take photos of a person to gererate sample data

Class_Data_Pickle.py #Turns your class data into a pickle for further formating

Random_Data_Pickle.py #Turns a few combined datasets of faces into a pickle for further formating

Pickle_Proccesing.py #Preposeses the pickles porduces by Random_Data_Pickle.py and Class_Data_Pickle.py and creates X.p and Y.p

Train.py #Trains a sequential model on X.p and Y.p

Application.py #Uses a camera connected to your computer and then uses the model generated from Train.py to predict

haarcascade_frontalface_default.xml #Used with cv2 in Class_Data_Create.py and Application.py


Dependencies: Tensorflow, Keras, opencv-python, Numpy
