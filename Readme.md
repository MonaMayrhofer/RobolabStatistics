# RobolabStatistics
This is more like an Opencv-Laboratory than an actual project. 

To see the whole definition (in German see) see: [Pflichtenheft](docs/pflichtenheft/Pflichtenheft.md)

# These programs use conda
To install the environment type

``conda env create -f environment.yml``

To update the environment in case of changes to ``environment.yml``

``conda env update -f environment.yml``

# Directories
## /apps
In this directory are some small programs, all are related to 
Computer-Vision

### /facedetector
This is an example of the Opencv Haarclassifier, trained with models from their git-repo. 
### /facepong
Pong. But in this version of pong you and your friends control the paddle
with your face.

### /plot_callback_test
A test-app for kerasplot.plot_callbacks.py from robolib

### /pymunktest
An example App to learn and understand pymunk. Pymunk was needd for facepong.

### /training_tf_simple_m
This directory contains a number of example-apps to understand neural networks and learn the api of several machine-learning-libraries like "Tensorflow","PyTorch","Keras"

### /ui_test
A test-app for robogui.pixel_editor from robolib.

### /facerecog
An app wich implements a siamese neural network in Pytorch and Keras. This forms the foundation of the finished Face-Detector.

## /legacyjava
We started with our Project using the Java language and we don't want to
throw it completely away. This is discontinued and code in there is not
bugfixed anymore. Don't expect changes in this directory.

## /robolib
Robolib is a collection of useful small libraries that are used by the apps. 

### /datamanager
A collection of utils to fetch training-data from external sources

### /images
A collection of utils to work with images

### /kerasplot
Callbacks for Keras to show the loss live in matplotlib plot

### /modelmanager
A collection of utils to fetch trained models from external sources. This is used for the Haar-Classifier of OpenCv

### /util
A collection of generic utils.