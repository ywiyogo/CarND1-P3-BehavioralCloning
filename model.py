# model.py
# Author: YWiyogo
# Descriṕtion: Model for training data

import tensorflow as tf
import numpy as np
import pandas
import pickle

from sklearn.model_selection import train_test_split

from keras.layers import Input, Flatten, Dense
from keras.models import Model

# Uploading the training data
# driving_log.csv: center_img, left_img, right_img, steering_angle, throttle, break, speed
# reading CSV file with different format can be done by using pandas lib.
# numpy.loattxt() can only read if the format of all data are the same
csv_file = "./TrainingData/driving_log.csv"
with open(csv_file, "r") as file:
    cols = ['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle',
            'break', 'speed']
    driving_data = pandas.read_csv(file, names=cols)
    print("Driving log: ", driving_data.shape)


# Input: images from the center camera of the car.
# Output: a new steering angle for the car.

# Split Training data to provide the validation data
X_train, X_valid, y_train, y_valid = train_test_split(driving_data['center_img'], driving_data['steering_angle'], test_size=0.2, random_state=0)
print("X_train: ", X_train.shape)
print("X_valid: ", X_valid.shape)

# Extend the data
# ---------------------------------
# Simulationg different position using multiple cameras

for i in range(driving_data.shape[0]):
   #create adjusted steering measurements for the side camera images
   steering_center = driving_data['steering_angle'][i]
   correction = 0.2 # this is a parameter to tune
   steering_left = steering_center + correction
   steering_right = steering_center - correction
   # add images and angles to data set
   X_train.extend(img_center, img_left, img_right)
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
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(43))
model.add(Activation('softmax'))