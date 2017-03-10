# **Behavioral Cloning Writeup Report** 

---

## Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/overfitting_e12_b230_s11918.png "Model Visualization"
[image2]: ./examples/overfitting_e15_b92_s10580.png "Grayscaling"
[image3]: ./examples/T1_MSE_e8_v30_b250.png "MSE validation 30"
[image4]: ./examples/T1_MSE_e7_v35_b250.png "MSE validation 35"
[image5]: ./examples/T1_MSE_e10_v35_b250.png "MSE validation 35"
[image6]: ./examples/T1_MSE_e10_v40_b250.png "MSE validation 40"
[image7]: ./examples/center_2017_02_27_23_31_47_738.jpg "Center Image"
[image8]: ./examples/flipped_img.png "Flipped Image"
[image9]: ./examples/T1_final_modelMSE.png "Final model MSE"
[image10]: ./examples/overborder1.png "Not drivable surface 1"


---
## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* helper_function.py containing the helper functions
* viewer.py for visualizing the history object of the Keras model
* model.h5.7z.001 and model.h5.7z.002 containing a trained convolution neural network. The size of model.h5 is 116 MB, thus I need to split it so that I can push it to github
* writeup_report.md summarizing the results
* video.mp4 containing the vidoeof the final result


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

After experimenting with a simple network with one fully connected layer and adapted imageNet model, I choose the NVidia end-to-end convolution network as the end model. The model consists of 3 convolution neural network with 5x5 filter sizes, 2 convolutional network with 3x3 filter size, and 5 .

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#### 2. Attempts to reduce overfitting in the model

My attemps to reduce overfitting are:
1. by inserting dropout layers in order to reduce overfitting,
2. by changing the parameter of the fully connected layers,
3. by tuning the batch size in my generator function,
4. by shuffle the input samples in the generator function,
5. by tuning the validation size,
6. by tuning the learning rate of the Adam optimizer

#### 3. Model parameter tuning

During my experiments, I was tuning these parameters:
1. Dropout rate
2. Number of fully connected layer
3. Batch size of the sample generator function
4. Size of the validation dataset in percent
5. Learning rate of the Adam optimizer 

The final parameters have these values:
1. Dropout rate: 0.2
2. FC 1164 -> 500 -> 100 -> 50 -> 10 -> 1
3. Batch size: 256
4. Validation size: 25% of the collected data points
5. Learning rate: 0.0005

By applying the dropout rate to 0.2, the learning rate less than 0.001 and adjusting the validation size to 35 %, it can delay the overfitting after 7th epoch. Too low or too high batch size can also affect the overfitting.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and more training data for the curving roads.

For details about how I created the training data, see the next section. 

---

### Model Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to obtain less loss value and validation loss value

My first step was to try a simple fully connected network. Then, I uses a convolution neural network model similar to the [imageNet model](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). I thought this model might be appropriate because it has the highest accuracy for the classification case. After my this experiment, the model consumes a lot of resources and very slow because of the high number of the convolution filter (more than 200).

The project description refers also to the [end-to-end learning convolution network](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) from NVIDIA autonomous car team. It showed me a better and more stable result than my previous simple model and the simplified ImageNet model. Thus, I started to tune the parameter of this NVIDIA convolution model.

My experiments show that changing the stride from 1 to 2 (or 2 to 1) and the filter size from 5x5 to 9x9 does not result a big difference in the resulting MSE loss.

To combat the overfitting, first I try to visualize the loss and the validation loss from the model by saving the keras model result as pickle data. Each time the model finishs the training, I visualize those value and decide if I need to try the model on the simulator or I just need to tune the paramaters. 

Afterwards, I experimented the model by increasing/lowering the batch size and the learning rate to reduce the overfitting. The number of epoch depends on these both parameters to avoid the overfitting. Compared to the second project which I used high numbers of epoch, in this project I observed that mostly I need a range between 4 and 8.

| Learning rate 0.001, batch size 230 | Learning rate 0.0008, batch size 32 |
|:----:|:----:|
| ![alt text][image1] | ![alt text][image2]|

My several experiments show that the vehicle will not drive properly if the difference between the MSE loss value of the training and the validation set is larger than approximately 0.04

| Validation 30% | Validation 35% | 
| :----: | :----: | 
| ![alt text][image3] | ![alt text][image4] |

| Validation 35% | Validation 40% | 
| :----: | :----: | 
| ![alt text][image5] | ![alt text][image6] |

I can observe that the overfitting happens normally after 5th epoch. Several experiments shows me that the `shuffle` function plays a significant role in this case.

The final step was to run the simulator to see how good the vehicle can drive around track one. There were a few spots where the vehicle fell off the track, I record new training data on that spot where the vehicle stuck or fell off the track.

However, the above model does not generate an acceptable result, since the vehicle can drive pop up onto ledges or over any surfaces that would otherwise be considered unsafe (see below figure). The simulation result was rejected.

| ![alt text][image10] 

Since more data sets did not provide me a better result in the simulation, I was starting to collect the data again. After I collected about 12131 samples, I observed a better result with the same model. There was some bad recording situation in my previous data sets.

At the end of the process, the vehicle is able to drive autonomously around the track 1 and a half of the track 2 without leaving the road. The bellow figure shows the final MSE loss values.

![alt text][image9]

#### 2. Final Model Architecture

The final model architecture (model.py lines 110-134) consisted of a convolution neural network with the following layers, layer sizes, and strides

| Conv. Layers | Number of Filter| Filter size | Stride |
| :----:|:----:|:----:|:----:|
|1|24| 5x5|(2, 2)|
|2|36| 5x5|(2, 2)|
|3|48| 5x5|(2, 2)|
|4|64| 3x3|(1, 1)|
|5|64| 3x3|(1, 1)|

I modify the full connected layer of the original Nvidia modelby adding a layer with 500 neurons. The other parameter stay the same. Thus, please refer to the [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

| FC Layers | Neurons |
| :----:| :----:|
|6|1164|
|7|500| 
|8|100| 
|9|50| 
|10|10| 
|11|1| 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![image center][image7]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center of the road. I recorded also more intensively at the secund and the third turning curves.

To augment the data set, I also flipped the images and the angles, thinking that this would generalize the learning. For example, here is an image that has then been flipped:

![alt text][image8]

I used also the images from the side cameras in order to get more data which steering angle is not zero.

After the collection process, I had 12131 number of data lines in CSV. Then, I randomly shuffled the data set and put 25% of the data into a validation set. 

I do not preprocess the images so intensive. The input images are cropped by the Keras platform with these parameter (top:50, bottom:20), (left: 10, right: 10).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by the above plot. I used an adam optimizer and I found out that 0.0005 was the ideal value for the learning rate at the beginning.

Based on my experiment, a good recording data sets are more significant than  the parameters tuning. Different data sets as the input of the neural network has different optimal parameters.

