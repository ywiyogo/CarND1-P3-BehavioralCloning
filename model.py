"""
Author: YWiyogo
Descriṕtion: Model for training data
"""
import numpy as np
import cv2
import csv
import pickle
import time
import sys
from sklearn.utils import shuffle

from helper_functions import get_relative_path
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Cropping2D, Convolution2D
from keras.layers import MaxPooling2D, Dropout, Lambda
from keras.models import Model, Sequential
from keras.optimizers import Adam


# Uploading the training data
# driving_log.csv: center_img, left_img, right_img, steering_angle, throttle,
# break, speed
# reading CSV file with different format can be done by using pandas lib.
# numpy.loattxt() can only read if the format of all data are the same
csv_file = "../p3_training_data/driving_log.csv"


def generator(samples, batch_size=32):
    """
    Generator function to create the image and augmented images, based on
    the lecture code.
    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            steer_angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(get_relative_path(batch_sample[0]))
                steering_center = float(batch_sample[3])
                if center_image is None:
                    print("Warning image is None object, cannot be flipped ",
                          get_relative_path(batch_sample[0]))
                    continue
                rand = 0
                # to avoid overfitting for angle 0, I utilize 3 random numbers
                # in order to replace the image either with the flipped or
                # left or right image
                if steering_center == 0.0:
                    rand = np.random.randint(3)
                    correction = 0.2  # this is a parameter to tune
                    if rand == 0:
                        images.append(center_image)
                        steer_angles.append(steering_center)
                    elif rand == 1:
                        steering_left = steering_center + correction
                        image_left = cv2.imread(get_relative_path(
                                                batch_sample[1]))
                        if image_left is not None:
                            images.append(image_left)
                            steer_angles.append(steering_left)
                        else:
                            print("Warning image right is None object")
                    else:
                        steering_right = steering_center - correction
                        image_right = cv2.imread(
                            get_relative_path(batch_sample[2]))
                        # add images and angles to data set
                        if image_right is not None:
                            images.append(image_right)
                            steer_angles.append(steering_right)
                        else:
                            print("Warning image right is None object")
                else:
                    # If angle !=0 choose either the center image or the
                    # flipped image
                    if np.random.randint(2) == 0:
                        images.append(center_image)
                        steer_angles.append(steering_center)
                    else:
                        # image flipping
                        image_flipped = np.fliplr(center_image)
                        meas_flipped = -steering_center
                        images.append(image_flipped)
                        steer_angles.append(meas_flipped)

            X_train_augmented = np.array(images)
            y_train_augmented = np.array(steer_angles)
            # print("X_train_augmented shape: ", X_train_augmented.shape)
            yield shuffle(X_train_augmented, y_train_augmented)

# -------------------------------------------
# Deep Learning Model Architectures with Keras
# -------------------------------------------
modelname = ""


def nvidia_net(dropout_rate, lrate=0.001):
    """
    Implement the nvidia CNN:
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    # channel, height, width = 3, 66, 200
    channel, height, width = 3, 160, 320
    print("Starting NVidia deep learning...")
    global modelname
    modelname = "nvidia"
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
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid',
                            activation="relu"))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid',
                            activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    adam_opt = Adam(lrate)
    model.compile(optimizer=adam_opt, loss="mse")
    # model.fit(X_train_augmented, y_train_augmented, validation_split=0.2,
    #           shuffle=True, nb_epoch=epoch)
    return model

# ImageNet like model:
# http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# It  contains  eight  learned  layers:
# five  convolutional  and  three  fully-connected.
#
def image_net(dropout_rate):
    """
    Adapted imageNet model with less filter
    """
    channel, height, width = 3, 160, 320
    global modelname
    modelname = "imagenet"
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
    model.add(Convolution2D(16, 5, 5, border_mode='valid', activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(20, 5, 5, border_mode='valid'))

    # FC
    model.add(Flatten())
    model.add(Dense(700))
    model.add(Dropout(dropout_rate))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.save('model_imagenet.h5')  # creates a HDF5 file 'my_model.h5'
    return model

# ------------------------------------------------------------------
# image from the simulator: 160x320 pixel
# Input: images from the center camera of the car.
# Output: a new steering angle for the car.

lines = []
with open(csv_file) as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)

print("Collected cvs dataset: ", len(lines))
train_samples, validation_samples = train_test_split(lines, test_size=0.20)
# My strategy is to generate each line with 2 samples
num_tsamples = len(train_samples)
num_vsamples = len(validation_samples)
batchsize = 200
epoch = int(sys.argv[1] if len(sys.argv) > 1 else 5)
train_generator = generator(train_samples, batchsize)
validation_generator = generator(validation_samples, batchsize)

print("Number of train samples: %dx2 = %d" % (len(train_samples), num_tsamples))
print("Number of validation samples: %dx2 = %d" % (len(validation_samples),
      num_vsamples))
print("Samples/epoch: ", num_tsamples)
print("Batch size: %d, epoch: %d" % (num_tsamples, epoch))

exec_model = nvidia_net(0.4, lrate=0.0005)

# Run the Keras model with generator
history_obj = exec_model.fit_generator(train_generator,
                                       samples_per_epoch=num_tsamples,
                                       validation_data=validation_generator,
                                       nb_val_samples=num_vsamples,
                                       nb_epoch=epoch)

# modelname = "model_" + modelname + "_e" + str(epoch) + "_b" + str(batchsize) \
#            + "_" + time.strftime("%Y%m%d_%H%M") + ".h5"
modelname = "model.h5"
exec_model.save(modelname)
print("Model saved in ", modelname)

# save to pickle in order to do the offline analysis
hist_data = history_obj.history
fname_hist = "hist_" + modelname + ".p"
pickle.dump(hist_data, open(fname_hist, "wb"))
