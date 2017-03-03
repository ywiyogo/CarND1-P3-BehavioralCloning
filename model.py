# model.py
# Author: YWiyogo
# Descriṕtion: Model for training data

import numpy as np
import cv2
import csv
import pandas
import pickle
from sklearn.utils import shuffle
from os.path import isfile
from helper_functions import show_histogram, get_relative_path
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Cropping2D, Convolution2D
from keras.layers import MaxPooling2D, Dropout, Lambda
from keras.models import Model, Sequential


# Uploading the training data
# driving_log.csv: center_img, left_img, right_img, steering_angle, throttle,
# break, speed
# reading CSV file with different format can be done by using pandas lib.
# numpy.loattxt() can only read if the format of all data are the same
csv_file = "../p3_training_data/driving_log.csv"


def generator(samples, batch_size=32):
    """
    Generator function to create the image and augmented images
    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steer_angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(get_relative_path(batch_sample[0]))
                steering_center = float(batch_sample[3])
                images.append(center_image)
                steer_angles.append(steering_center)

                if steering_center != 0.0:
                    # Example of image flipping
                    if center_image is not None:
                        image_flipped = np.fliplr(center_image)
                        meas_flipped = -steering_center
                        images.append(image_flipped)
                        steer_angles.append(meas_flipped)
                    else:
                        print("Warning image is None object, cannot be flipped")
                    # Using side cameras
                    correction = 0.2  # this is a parameter to tune
                    steering_left = steering_center + correction
                    image_left = cv2.imread(get_relative_path(batch_sample[1]))
                    if image_left is not None:
                        images.append(image_left)
                        steer_angles.append(steering_left)
                    else:
                        print("Warning image right is None object")

                    steering_right = steering_center - correction
                    image_right = cv2.imread(get_relative_path(batch_sample[2]))
                    # add images and angles to data set
                    if image_right is not None:
                        images.append(image_right)
                        steer_angles.append(steering_right)
                    else:
                        print("Warning image right is None object")

            X_train_augmented = np.array(images)
            y_train_augmented = np.array(steer_angles)
            yield shuffle(X_train_augmented, y_train_augmented)


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
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print("Number of train samples: ", len(train_samples))
print("Number of validation samples: ", len(validation_samples))
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#show_histogram(y_train_augmented, "Histogram of the provided training data set")

# -------------------------------------------
# Deep Learning Model Architectures with Keras
# -------------------------------------------

def simple_regression_network(epoch=2):
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
    model.fit(X_train_augmented, y_train_augmented, validation_split=0.2,
              shuffle=True, nb_epoch=epoch)
    modelname = "model1_" + str(epoch) + ".h5"
    model.save(modelname)
    print("Model saved in ", modelname)
    return model


def nvidia_net(epoch=5):
    """
    Implement the nvidia CNN:
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    # channel, height, width = 3, 66, 200
    channel, height, width = 3, 160, 320
    print("Starting NVidia deep learning...")
    model = Sequential()
    # cropping image from (up, down), (left,right)
    model.add(Cropping2D(cropping=((50, 20), (10, 10)),
                         input_shape=(height, width, channel)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid',
                            activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid',
                            activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid',
                            activation="relu"))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid',
                            activation="relu"))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid',
                            activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    # model.fit(X_train_augmented, y_train_augmented, validation_split=0.2,
    #           shuffle=True, nb_epoch=epoch)
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples), nb_epoch=epoch)

    model.save('model_nvidia.h5')  # creates a HDF5 file 'my_model.h5'
    print("Model saved in model_nvidia.h5")
    return model
# I use ImageNet like model:
# http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# It  contains  eight  learned  layers:
# five  convolutional  and  three  fully-connected.


def image_net(epoch=5):
    """
    Adated imageNet model
    """
    dropout_rate = 0.4
    channel, height, width = 3, 160, 320
    model = Sequential()
    # crop the images
    # set up cropping2D layer 160x320 -> 97x305 pixels
    model.add(Cropping2D(cropping=((43, 20), (7, 8)),
                         input_shape=(height, width, channel)))
    # Normalize
    # model.add(Lambda(lambda x: x / 128 - 1.,
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
    model.add(Convolution2D(20, 5, 5, border_mode='valid'))
    # output: 5x68x384

    # FC
    model.add(Flatten())
    # model.add(Dense(7000))
    model.add(Dense(700))
    model.add(Dropout(dropout_rate))
    model.add(Dense(50))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_augmented, y_train_augmented, validation_split=0.2,
              shuffle=True, nb_epoch=epoch)

    model.save('model_imagenet.h5')  # creates a HDF5 file 'my_model.h5'
    return model


# > python drive.py model.h5 <output_img_dir>

#simple_regression_network(3)
nvidia_net()