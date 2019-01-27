# Image-Super-Resolution-Convolutional-Neural-Network
Python implementation of image super resolution convolutional neural network

This script reproduces the Super Resolution Convolutional Neural Network (SRCNN) as described in the paper
'Image Super Resolution Using Deep Convolutional Networks' by C. Dong, C. C. Loy, K. He, X. Tang (2015). 

The main differences lie in the learning rate schedule and the loss optimizer used in the SRCNN model.
In addition, this script also includes functions which are used to preprocess the input images into a proper format
so that they can be used to train the SRCNN model.

Requirements:
- cv2
- gc
- glob
- numpy
- keras (tensorflow backend)
- gdal
