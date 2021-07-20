import cvlib as cv
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
# from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
import random

epochs = 30
lr = 10**random.uniform(-3,-4)
batch_size = 32

aug = ImageDataGenerator(rotation_range = 25, width_shift_range = 0.1,
                         height_shift_range=0.1, shear_range = 0.2, zoom_range = 0.2,
                         horizontal_flip = True, fill_mode= "nearest")


def capture_frames():
    cam = cv2.VideoCapture(0)

    count = 0

    while cam.isOpened():

        # if count == 100:
        #    break
        count += 1

        status, frame = cam.read()
        print(type(frame))
        face, confidence = cv.detect_face(frame)

        for idx, f in enumerate(face):

            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            face_crop = np.copy(frame[startY:endY, startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                if count >= 1000:
                    break
                continue

            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            # print(f)
            # cv2.imshow("face",face)
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            user = "u1"
            fpath = "E:/data/" + user + "." + str(count) + ".jpg"
            cv2.imwrite(fpath, frame)

            # model.predict(face_crop)[0]

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # cv2.putText(frame,(startX,Y), cv2.FONT_HERSHEY_SIMPLEX,
            #           0.7,(0,255,0),2)

        cv2.imshow("Face recognition", frame)

        if cv2.waitKey(1) == 13 or count == 1000:
            break

    cam.release()
    cv2.destroyAllWindows()

def print_2():
    print(2)

def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('sigmoid'))
    model.summary()

    return model

def label(img_name):
    name = img_name.split('.')[0].split(os.path.sep)
    name = name[-1]
    #print(name)
    if name == "u1":
        return np.array([1,0])
    else:
        return np.array([0,1])


def recognize(nic_img_nparr, frames): # nic_img should be a np array
    nic_img= nic_img_nparr
    data = []
    labels = []
    img_dims = (96, 96, 1)

    img_files = [f for f in
                 glob.glob(r'C:\Users\HP\data' + "/**/*", recursive=True)]  # if not os.path.isdir('/usr/bin')]
    random.shuffle(img_files)

    img_files = img_files + frames

    print('Fetching Data')
    # FACE DETECTION
    for img in img_files:
        r_image = cv2.imread(img)
        face = cv.detect_face(r_image)

        if face:
            f = face[0]

            image = cv2.cvtColor(r_image, cv2.COLOR_BGR2GRAY)
            if f:
                f = f[0]

                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                image = np.copy(image[startY:endY, startX:endX])

                if image.size:
                    image = cv2.resize(image, (img_dims[0], img_dims[1]))
                    img_2 = Image.fromarray(image)
                    image = img_to_array(image)

                    data.append(image)
                    labels.append(label(img))

    data = np.asarray(data)
    labels = np.asarray(labels)
    print('data fetched')
    opt = Adam(lr=lr, decay=lr / epochs)

    model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.02)

    H = model.fit(aug.flow(x_train, y_train, batch_size=batch_size),
                  steps_per_epoch=len(x_train) / batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  )
    # cv2.imshow('nic',n_img)
    n= cv2.imwrite('n.jpg',nic_img)
    x = cv2.imread('n.jpg')
    fe = cv.detect_face(x)

    s = fe[0]
    print(fe)
    # print(type(s))
    s = s[0]
    # print(s[0])
    (startX, startY) = s[0], s[1]
    (endX, endY) = s[2], s[3]

    # cv2.rectangle(frame,(startX,startY), (endX,endY),(0,255,0),2)

    face_crop = np.copy(x[startY:endY, startX:endX])
    img_2 = Image.fromarray(face_crop)  # .save('img.jpg')

    # if(face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
    #    continue

    face_crop = cv2.resize(face_crop, (96, 96))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    # face_crop = face_crop.astype("float")/255.0
    # print(f)
    # cv2.imshow("face",face_crop)
    img_2 = Image.fromarray(face_crop)  # .save('i.jpg')
    # plt.imshow(face_crop)

    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    con = model.predict(face_crop)[0]
    # print(con)
    print(con[0] * 100, "%")
    return con[0]