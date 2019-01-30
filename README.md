# Image-Super-Resolution-Convolutional-Neural-Network-for-Remote-Sensing
Python implementation of image super resolution convolutional neural network

This script reproduces the Super Resolution Convolutional Neural Network (SRCNN) as described in the paper
'Image Super Resolution Using Deep Convolutional Networks' by C. Dong, C. C. Loy, K. He, X. Tang (2015). 

The main differences lie in the learning rate schedule and the loss optimizer used in the SRCNN model.
In addition, this script also includes functions which are used to preprocess the input images into a proper format
so that they can be used to train the SRCNN model, and functions which uses the model to create an upsampled version of
a raster file, and write it to file.

NOTE: gdal would be unable to write a very large array to file, so please do try to keep the input image file as small as possible.
      (5000 x 5000 x 4 arrays can still be written to file)

Requirements:
- cv2
- gc
- glob
- numpy
- keras (tensorflow backend)
- gdal
