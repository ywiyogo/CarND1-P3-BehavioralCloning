#**Behavioral Cloning Writeup Report** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/overfitting_e12_b230_s11918 "Model Visualization"
[image2]: ./examples/overfitting_e15_b230_s10580.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

YW: Using the first simple model from the lecture (one layer), I observed a huge different between normalized images using lambda and without normalization.


After experimenting with a simple model and adapted imageNet model, I choose the NVidia end-to-end convolution network as the end model. The model consists of 3 convolution neural network with 5x5 filter sizes, 2 convolutional network with 3x3 filter size, and 5  (model.py lines 18-24).

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

My attemps to reduce overfitting are:
1. by inserting dropout layers in order to reduce overfitting (model.py lines 21). 
2. by adding one fully connected layer
3. by applying higher batch size in my generator function
4. by shuffle the input samples in the generator function

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and more training for curving roads

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to obtain less loss value and validation loss value

My first step was to use a convolution neural network model similar to the imageNet model. I thought this model might be appropriate because it has the highest accuracy. After my first experiment, the model consumes a lot of resources and very slow because of the high number of the convolution filter (more than 200).

From the project link and the discussions in forum, I started to begin with the end-to-end learning convolution network from NVIDIA autonomous car team.

To combat the overfitting, first I try to visualize the loss and the validation loss from the model by saving the keras model result as pickle data. Each time the model finishs the training, I visualize those value and decide if I need to try the model on the simulator or I just need to tune the paramaters.

With the model is fixed, I can increase the batch size and lowering the learning rate to reduce the overfitting. This approach has to be compensated by increasing the epoch. Compared to the second project which I used high numbers of epoch, in this project I observed that mostly if epoch is more than 10, the training is going to be overfitting.

![alt text][image1]
![alt text][image2]

After I collected more than 11000 samples, I can observe that the overfitting happens normally after 5th epoch. Several experiments shows me that the `shuffle` function plays a significant role in this case.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I record new training data on the spot where the vehicle stuck or fell off the track.

At the end of the process, the vehicle is able to drive autonomously around the track 1 and track 2 without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Since I only add one full connected layer to the Nvidia model, please refers the visualization to this [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center of the road. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would generalize the learning. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had 14898 number of data points. Then, I cut the data points to 14000. Afterwards, I preprocessed this data by cropping the border (top, bottom, left, right) in order to eliminate redundant information.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer and I was tuning the learning rate.
