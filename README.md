# Project 3: Use Deep Learning to Clone Driving Behavior

### Solution Design Approach

The overall strategy for deriving a model architecture was to obtain less loss value and validation loss value

My first step was to try a simple fully connected network. Then, I uses a convolution neural network model similar to the [imageNet model](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). I thought this model might be appropriate because it has the highest accuracy for the classification case. After my this experiment, the model consumes a lot of resources and very slow because of the high number of the convolution filter (more than 200).

The project description refers also to the [end-to-end learning convolution network](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) from NVIDIA autonomous car team. It showed me a better and more stable result than my previous simple model and the simplified ImageNet model. Thus, I started to tune the parameter of this NVIDIA convolution model.

My experiments show that changing the stride from 1 to 2 (or 2 to 1) and the filter size from 5x5 to 9x9 does not result a big difference in the resulting MSE loss.

To combat the overfitting, first I try to visualize the loss and the validation loss from the model by saving the keras model result as pickle data. Each time the model finishs the training, I visualize those value and decide if I need to try the model on the simulator or I just need to tune the paramaters. 

Afterwards, I experimented the model by increasing/lowering the batch size and the learning rate to reduce the overfitting. The number of epoch depends on these both parameters to avoid the overfitting. Compared to the second project which I used high numbers of epoch, in this project I observed that mostly I need a range between 5 and 10.

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

After I collected more than 11000 samples, I can observe that the overfitting happens normally after 5th epoch. Several experiments shows me that the `shuffle` function plays a significant role in this case.

The final step was to run the simulator to see how good the vehicle can drive around track one. There were a few spots where the vehicle fell off the track, I record new training data on that spot where the vehicle stuck or fell off the track.

At the end of the process, the vehicle is able to drive autonomously around the track 1 and a half of the track 2 without leaving the road.

### Final Model Architecture

The final model architecture (model.py lines 110-134) consisted of a convolution neural network with the following layers, layer sizes, and strides

| Conv. Layers | Number of Filter| Filter size | Stride |
| :----:|:----:|:----:|:----:|
|1|24| 5x5|(2, 2)|
|2|36| 5x5|(2, 2)|
|3|48| 5x5|(2, 2)|
|4|64| 3x3|(1, 1)|
|5|64| 3x3|(1, 1)|

I modify the full connected layer of the original Nvidia model. Instead of densing the size to 1164 neurons, I dense to 500 neurons. The other parameter stay the same. Thus, please refer to the [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

| FC Layers | Neurons |
| :----:| :----:|
|6|500| 
|7|100| 
|8|50| 
|9|10| 
|10|1| 

### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![image center][image7]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center of the road. Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped the images and the angles, thinking that this would generalize the learning. For example, here is an image that has then been flipped:

![alt text][image8]

I used also the images from the side cameras in order to get more data which steering angle is not zero.

After the collection process, I had 13098 number of data lines in CSV. Then, I randomly shuffled the data set and put 35% of the data into a validation set. 

I do not preprocess the images so intensive. The input images are cropped by the Keras platform with these parameter (top:50, bottom:20), (left: 10, right:10).


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the above plot. I used an adam optimizer and I found out that 0.0005 was the ideal value aftertuning the learning rate.
