# Project 3 - face recognition
import os

import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import cv2
from sklearn.model_selection import train_test_split




def loadData():
    facePath = "data"
    noFacePath = "dataNoFace"
    X = []
    y = []
    for name in os.listdir(facePath):
        X.append(cv2.imread(os.path.join(facePath, name)))
    for name in os.listdir(noFacePath):
        img = cv2.imread(os.path.join(noFacePath, name))
        img = cv2.resize(img, (178, 218))
        X.append(img)

    X = np.array(X)
    y = np.zeros(X.shape)
    n_faces = len(X)
    y[:n_faces] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

    print(y)

def train(X_train, y_train, X_test, y_test):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Ocena modelu
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main():
    print("Hello")

    X_train, X_test, y_train, y_test = loadData()
    #train(X_train, y_train, X_test, y_test)



if __name__ == '__main__':
    main()