# model.py
# Author: YWiyogo
# Descriṕtion: Model for training data

import numpy as np
import csv, cv2
import pandas
import pickle
from os.path import isfile
from helper_functions import show_histogram, get_relative_path
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Input, Cropping2D, Convolution2D
from keras.layers import MaxPooling2D, Activation, Dropout, Lambda
from keras.models import Model, Sequential


# Uploading the training data
# driving_log.csv: center_img, left_img, right_img, steering_angle, throttle,
# break, speed
# reading CSV file with different format can be done by using pandas lib.
# numpy.loattxt() can only read if the format of all data are the same
csv_file = "../p3_training_data/driving_log.csv"

if not isfile('augmented_data.p'):
    lines = []
    with open(csv_file) as file:
        # cols = ['center_img', 'left_img', 'right_img', 'steering_angle',
        #         'throttle', 'break', 'speed']
        # driving_data = pandas.read_csv(file, names=cols)
        # print("Driving log: ", driving_data.shape)
        reader = csv.reader(file)
        for line in reader:
            lines.append(line)
# image from the simulator: 160x320 pixel
# Input: images from the center camera of the car.
# Output: a new steering angle for the car.

    images = []
    measurements = []
    for line in lines:
        curr_path = get_relative_path(line[0])
        image = cv2.imread(curr_path)
        if image is not None:
            images.append(image)
            steer_angle = float(line[3])
            measurements.append(steer_angle)
        else:
            print("Warning: image data is empty! Check the filename!")

    X_train = np.array(images)
    y_train = np.array(measurements)
    # Split Training data to provide the validation data
    # X_train, X_valid, y_train, y_valid = train_test_split(
    #     driving_data['center_img'], driving_data['steering_angle'], test_size=0.2,
    #     random_state=0)
    print("X_train: ", X_train.shape)
    #print("X_valid: ", X_valid.shape)

    show_histogram(y_train, "Histogram of the provided training \
    #               data set")
    # Extend the data
    # ---------------------------------
    # Simulationg different position using multiple cameras
    augmented_imgs, augmented_meas = [], []

    for i in range(len(lines)):
        # create adjusted steering measurements for the side camera images
        steering_center = float(lines[i][3])
        # Example of image flipping
        image_flipped = np.fliplr(images[i])
        meas_flipped = -measurements[i]
        augmented_imgs.append(image_flipped)
        augmented_meas.append(meas_flipped)

        correction = 0.2  # this is a parameter to tune
        steering_left = steering_center + correction
        path_left = get_relative_path(lines[i][1])
        image_left = cv2.imread(path_left)
        augmented_imgs.append(image_left)
        augmented_meas.append(steering_left)

        steering_right = steering_center - correction
        path_right = get_relative_path(lines[i][2])
        image_right = cv2.imread(path_right)
        # add images and angles to data set
        augmented_imgs.append(image_right)
        augmented_meas.append(path_right)

    X_train_augmented = np.concatenate((X_train, augmented_imgs), axis=0)
    y_train_augmented = np.concatenate((y_train, augmented_meas), axis=0)
    # print("Dataset after extending the diffrent position using multiple camera")
    print("y_train_augmented: ", X_train_augmented.shape)


    #save to pickle
    tdata = {}
    tdata['features'] = X_train_augmented
    tdata['measurements'] = y_train_augmented
    pickle.dump(tdata, open("augmented_data.p", "wb"))

else:
    print("Augmented data found")
    with open('augmented_data.p', 'rb') as f:
        data = pickle.load(f)
        # TODO: Load the feature data to the variable X_train
        X_train_augmented = data['features']
        # TODO: Load the label data to the variable y_train
        y_train_augmented = data['measurements']

show_histogram(y_train, "Histogram of the provided training\
                   data set")

def simple_regression_network(epoch=2 ):
    """
    Simple Regression network for testing
    """
    channel, height, width = 3, 160, 320
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5,
                     input_shape=(height, width, channel)))
    model.add(Cropping2D(cropping=((60, 25), (7, 8))))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train_augmented, y_train_augmented, validation_split=0.2, shuffle=True, nb_epoch=epoch)
    modelname = "model1_" + str(epoch) + ".h5"
    model.save(modelname)
    return model

# -------------------------------------------
# Deep Learning Model Architecture with Keras
# -------------------------------------------
# I use ImageNet like model:
# http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# It  contains  eight  learned  layers:
# five  convolutional  and  three  fully-connected.


def imageNet(epoch=5):
    dropout_rate = 0.4
    channel, height, width = 3, 160, 320
    model = Sequential()
    # crop the images
    # set up cropping2D layer 160x320 -> 97x305 pixels
    model.add(Cropping2D(cropping=((43, 20), (7, 8)),
                         input_shape=(height, width, channel)))
    # Normalize
    #model.add(Lambda(lambda x: x / 128 - 1.,
    #                 input_shape=(channel, 97, 305),
    #                 output_shape=(channel, 97, 305)))
    # 1 conv filter 11x11x3, stride: 2:
    # height: (97-11)/2 +1 = 44; weight: (305-11)/2 +1= 148
    model.add(Convolution2D(6, 5, 5, subsample=(2, 2), border_mode='valid',
                            activation="relu"))
    model.add(MaxPooling2D())
    # Output: 22x74x96

    # conv2 256 filter 5x5x48, stride: 1
    # height: (22-5)/1 + 1= 18, (148-5)+1 = 144
    model.add(Convolution2D(16, 5, 5, border_mode='valid', activation="relu"))
    model.add(MaxPooling2D())
    # Output: 9x72x256

    # 3rd conv 384 filter 5x5x256, stride: 1
    # h: 9-5+1= 5, w: 72-5+1= 68
    #model.add(Convolution2D(384, 5, 5, border_mode='valid'))
    # output: 5x68x384

    # FC
    model.add(Flatten())
    # model.add(Dense(7000))
    # model.add(Dropout(dropout_rate))

    model.add(Dense(500))
    model.add(Dense(50))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_augmented, y_train_augmented, validation_split=0.2,
              shuffle=True, nb_epoch=epoch)

    model.save('model_imagenet.h5')  # creates a HDF5 file 'my_model.h5'
    return model


def end2endCNN():
    model = Sequential()
    return model

# > python drive.py model.h5 <output_img_dir>

#simple_regression_network(3)
#imageNet()