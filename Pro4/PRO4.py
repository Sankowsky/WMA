# Project 4 - face recognition
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


import cv2
import numpy as np





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
    y = np.zeros(len(X))
    n_faces = len(X)
    y[:n_faces] = 1
    print(y)
    #y = np.reshape(y, (len(y), 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

    print(y)

def train(X_train, y_train, X_test, y_test):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(218, 178, 3), activation='relu'))    #pytanie do tego
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=1)

    # Ocena modelu
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main():
    print("Hello")
    showFaces()

    #X_train, X_test, y_train, y_test = loadData()
    #train(X_train, y_train, X_test, y_test)




def showFaces():
    #global swich_lib, fun, img
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

            img = frame
            for (x, y, w, h) in faces:
                roi = img[y:y + h, x:x + w]  # Region of interest
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 50, 250), 2)
            cv2.imshow("faces", img)

        key = cv2.waitKey(1)

        if key == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()