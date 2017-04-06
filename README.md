# Driving Behavioral Cloning with Deep Learning
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## About this repository

This repository is a project that simulates a behavioral cloning for autonomous driving using the deep learning approach.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Requirements

1. Install the Udacity starting kit from <https://github.com/udacity/CarND-Term1-Starter-Kit>

2. Get the Udacity simulator:
   * [For Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
   * [For Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)

## Installation

    git clone https://github.com/ywiyogo/CarND1-P3-BehavioralCloning.git

## Usage

1. Merge the model.h5.7z.001, model.h5.7z.002, model.h5.7z.003 containing a trained convolution neural network to model.h5

2. Go to the repository folder and activate the virtualenvironment:

        source activate carnd-term1

3. Start the deep learning 

        python drive.py model.h5


4. Start the simulator program and choose the "Autonomous Mode"

5. Optional: to record a video of the simulation, run the 3rd command above with an output directory name as the second argument:

        python drive.py model.h5 <output_dir_name>
        
    After the simulation convert the images in the output directory to a video by executing this command:
    
        python video.py <output_dir_name> --fps 48

## Documentation
Please read this [writeup](https://github.com/ywiyogo/CarND1-P3-BehavioralCloning/blob/master/writeup_report.md)

