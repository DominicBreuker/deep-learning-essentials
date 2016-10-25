# Deep Learning Essentials

Deep learning essentials is a collection of reusable code examples to get you started with deep learning.
The deep learning library of choice is Keras.
All the code works with both Theano and Tensorflow as backend.
Below is an overview of what you will find in this repository.
Each folder contains it's own readme with detailed instructions.

## Disclaimer

This is work in progress and currently extremely unorganized.
Use at your own risk.
You are for sure encountering lots of obstacles in getting started ;)

## Overview of content

### images

Any code specific to image processing.

#### VGG 16

Keras model for the VGG 16 architecture with pre-trained weights.
Comes with code to load images with proper preprocessing.

#### DeepDrive

Port of [deepdrive.io](http://deepdrive.io/).
Keras model for a CNN able to keep a car in  the lane.
Pretrained weights available.
To get started, go to root path of repo and do:

```
export PYTHONPATH=$(pwd)
KERAS_BACKEND=theano python images/pretrained/deep_drive/model_test.py
KERAS_BACKEND=tensorflow python images/pretrained/deep_drive/model_test.py
```

This should successfully predict 6 values each for 6 test images.
If things work, you should see losses around 0.01.
