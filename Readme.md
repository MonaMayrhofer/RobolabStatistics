# RobolabStatistics
This is more like an Opencv-Laboratory than an actual project. 

To see the whole definition (in German see) see: [Pflichtenheft](docs/pflichtenheft/Pflichtenheft.md)

# These programs use conda
To install the environment type

``conda env create -f environment.yml``

To update the environment in case of changes to ``environment.yml``

``conda env update -f environment.yml``

# Directories
## apps
In this directory are some small programs, all are related to 
Computer-Vision

### Facedetector
A small sample project in which the Opencv-Haar-Classifier is tested.

### Facepong
Pong. But in this version of pong you and your friends control the paddle
with your face.

## legacyjava
We started with our Project using the Java language and we don't want to
throw it completely away. This is discontinued and code in there is not
bugfixed anymore. Don't expect changes in this directory.

## robolib
Robolib is a collection of useful small libraries that are used by the apps. 