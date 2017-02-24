# model.py
# Author: YWiyogo
# Descriṕtion: Model for training data

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Input, Cropping2D, Convolution2D
from keras.layers import MaxPooling2D, Activation
from keras.models import Model, Sequential


# Uploading the training data
# driving_log.csv: center_img, left_img, right_img, steering_angle, throttle,
# break, speed
# reading CSV file with different format can be done by using pandas lib.
# numpy.loattxt() can only read if the format of all data are the same
csv_file = "./TrainingData/driving_log.csv"
with open(csv_file, "r") as file:
    cols = ['center_img', 'left_img', 'right_img', 'steering_angle',
            'throttle', 'break', 'speed']
    driving_data = pandas.read_csv(file, names=cols)
    print("Driving log: ", driving_data.shape)

# image from the simulator: 160x320 pixel
# Input: images from the center camera of the car.
# Output: a new steering angle for the car.

# Split Training data to provide the validation data
X_train, X_valid, y_train, y_valid = train_test_split(
    driving_data['center_img'], driving_data['steering_angle'], test_size=0.2,
    random_state=0)
print("X_train: ", X_train.shape)
print("X_valid: ", X_valid.shape)

# Extend the data
# ---------------------------------
# Simulationg different position using multiple cameras

for i in range(driving_data.shape[0]):
    # create adjusted steering measurements for the side camera images
    steering_center = driving_data['steering_angle'][i]
    correction = 0.2  # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    # add images and angles to data set
    X_train.extend(driving_data['center_img'][i], driving_data['left_img'][i],
                   driving_data['right_img'][i])
    y_train.extend(steering_center, steering_left, steering_right)

print("Dataset after extending the diffrent position using multiple camera")
print("X_train: ", X_train.shape)
print("X_valid: ", X_valid.shape)

# Example of image flipping
# image_flipped = np.fliplr(image)
# measurement_flipped = -measurement

# -------------------------------------------
# Deep Learning Model Architecture with Keras
# -------------------------------------------
# I use ImageNet like model:
# http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# It  contains  eight  learned  layers:
# five  convolutional  and  three  fully-connected.

dout_rate = 0.4
model = Sequential()
# crop the images
# set up cropping2D layer 160x320 -> 90x300 pixels
model.add(Cropping2D(cropping=((50, 20), (10, 10)), input_shape=(160, 320, 3)))
# 1 conv filter 11x11x3, stride: 4
model.add(Convolution2D(32, 3, 3, input_shape=(111, 300, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(dout_rate))
model.add(Activation('relu'))
# 1 conv filter 5x5xn, stride: 2


# 1 conv filter 5x5xm, stride: 2


# 1 conv filter 3x3xm, stride:1

# FC
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(43))
model.add(Activation('softmax'))


model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# > python drive.py model.h5 <output_img_dir>
