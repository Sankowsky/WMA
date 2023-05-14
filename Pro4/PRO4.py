# Project 4 - face recognition

import os
import re

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.utils import to_categorical


def loadData():
    global path, names
    X = []
    y = []
    names = {"inna": 0}
    #names = {0 : "inna"}


    for file in os.listdir(path):
        name = file.split('_')[0]

        if name not in names:
            n = max(names.values())
            names[name] = n+1

        X.append(cv2.imread(os.path.join(path, file)))
        y.append(names.get(name))


    #print((len(x), y_) for x, y_ in zip(X, y))
    #y = np.reshape(y, (len(y), 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test


def train(X_train, y_train, X_test, y_test):
    global model
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    #y_test = to_categorical(y_test)
    #y_train = to_categorical(y_train)

    print(len(X_train))
    print(len(y_train))

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1, activation='softmax'))

    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
    model.save('model_fit.h5')
    model = keras.models.load_model('model_fit.h5')

    # model = Sequential()
    #
    # model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Flatten())
    #
    # model.add(Dense(units=128, activation='relu'))
    # model.add(Dense(units=1, activation='sigmoid'))
    #
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #
    # model.fit(X_train, y_train, epochs=2, batch_size=5)

    # Ocena modelu
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def showFaces():
    global createFace, names, model
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                roi = cv2.resize(frame[y:y + h, x:x + w], (200, 200))  # Region of interest
                if createFace:
                    createDataSet(roi)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 50, 250), 2)

                resized_frame = cv2.resize(frame, (200, 200))
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                input_image = np.expand_dims(resized_frame, axis=0)
                name = model.predict(input_image)
                print(int(name[0][0]))
                name = [k for k, v in names.items() if v == int(name[0][0])][0]

                frame = cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("faces", frame)
        key = cv2.waitKey(1)

        if key == 27:
            cv2.destroyAllWindows()
            break


def createDataSet(image):
    print(f"Tworzę osobę {name}")

    files = [f for f in os.listdir(path) if re.search(f"{name}_\d+\.jpg", f)]
    n = 0
    if files:
        n = max([int(f.split('_')[-1].split('.')[0]) for f in files])
    out_name = f"{name}_{n+1}.jpg"

    cv2.imwrite(f"{path}{out_name}", image)


def main():
    global createFace, name, path
    path = "./faces/"
    createFace= False


    if createFace:
        name = "inna" #input("Wpisz nazwę osoby wczytywanej: ")



    X_train, X_test, y_train, y_test = loadData()
    train(X_train, y_train, X_test, y_test)
    showFaces()

if __name__ == '__main__':
    main()