import cv2
import gc
import glob
import numpy as np
from keras.models import Input, Model
from keras.layers import Conv2D
from keras.optimizers import Adam
from osgeo import gdal



def blurred_image_creation(img, n_factor):
    """ 
    This function generates the blurred version of the original input image, so as to create the low resolution image
    used for Super - Resolution Convolutional Neural Network (SRCNN) model training. 
    
    Inputs:
    - img: Numpy array of the original image which is to be used for SRCNN model training
    - n_factor: The upscaling factor by which the SRCNN model should be trained to upscale
    
    Outputs:
    - blurred_img_sam: Numpy array of Gaussian blurred version of original input image for use as training data for 
                       SRCNN model
    
    """
    
    blurred_img = np.zeros((img.shape))
    for i in range(img.shape[2]):
        blurred_img[:, :, i] = cv2.GaussianBlur(img[:, :, i], (5, 5), 0)
    blurred_img_small = cv2.resize(blurred_img, (int(img.shape[1] / n_factor), int(img.shape[0] / n_factor)), 
                                   interpolation = cv2.INTER_AREA)
    blurred_img_sam = cv2.resize(blurred_img_small, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
    
    return blurred_img_sam



def image_clip_to_segment(image_array, blurred_image_array, image_height_size, image_width_size, mode, n_bands, 
                         f1, f2, f3, padding):
    """ 
    This function is used to cut up original input images of any size into segments of a fixed size, with empty clipped areas 
    padded with zeros to ensure that segments are of equal fixed sizes and contain valid data values. The function then 
    returns a 4 - dimensional array containing the entire original input image and its corresponding blurred image in the 
    form of fixed size segments as training data inputs for the SRCNN model.
    
    Inputs:
    - image_array: numpy array of original input image to be used for SRCNN model training
    - blurred_image_array: numpy array of Gaussian blurred version of original input image to be used for SRCNN model training
    - image_height_size: Height of image to be fed into the SRCNN model for training
    - image_width_size: Width of image to be fed into the SRCNN model for training
    - mode: Integer representing the status of height and width of input image which is defined under 'image_array'
    - n_bands: Number of channels contained in the input image
    - f1: size of kernel to be used for the first convolutional layer
    - f2: size of kernel to be used for the second convolutional layer
    - f3: size of kernel to be used for the last convolutional filter
    - padding: String which determines whether the input image is padded during model training so as to maintain the original image 
               size ('same') or to use the clipped version as implemented in the paper ('original')
    
    Output:
    - blurred_segment_array: 4 - Dimensional numpy array of Gaussian blurred image to serve as training data for SRCNN model
    - image_segment_array: 4 - Dimensional numpy array of original input image to serve as target data for training SRCNN 
                           model
    
    """
    
    x_size = ((image_array.shape[0] // image_height_size) + 1) * image_height_size
    y_size = ((image_array.shape[1] // image_width_size) + 1) * image_width_size
    
    if mode == 0:
        img_complete = np.zeros((x_size, image_array.shape[1], n_bands))
        blurred_complete = np.zeros((x_size, image_array.shape[1], n_bands))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : n_bands] = image_array
        blurred_complete[0 : blurred_image_array.shape[0], 0 : blurred_image_array.shape[1], 0 : n_bands] = blurred_image_array
    elif mode == 1:
        img_complete = np.zeros((image_array.shape[0], y_size, n_bands))
        blurred_complete = np.zeros((image_array.shape[0], y_size, n_bands))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : n_bands] = image_array
        blurred_complete[0 : blurred_image_array.shape[0], 0 : blurred_image_array.shape[1], 0 : n_bands] = blurred_image_array
    elif mode == 2:
        img_complete = np.zeros((x_size, y_size, n_bands))
        blurred_complete = np.zeros((x_size, y_size, n_bands))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : n_bands] = image_array
        blurred_complete[0 : blurred_image_array.shape[0], 0 : blurred_image_array.shape[1], 0 : n_bands] = blurred_image_array
    elif mode == 3:
        img_complete = image_array
        blurred_complete = blurred_image_array
        
    img_list = []
    blurred_list = []
    
    start_index = int(((f1 - 1) / 2) + ((f2 - 1) / 2) + ((f3 - 1) / 2))
    
    for i in range(0, image_array.shape[0], image_height_size):
        for j in range(0, image_array.shape[1], image_width_size):
            img_orig = img_complete[i : i + image_height_size, j : j + image_width_size, 0 : n_bands]
            if padding = 'original':
                img_list.append(img_orig[start_index : (image_height_size - start_index), 
                                         start_index : (image_width_size - start_index), 0 : n_bands])
            else:
                img_list.append(img_orig)
            blurred_list.append(blurred_complete[i : i + image_height_size, j : j + image_width_size, 0 : n_bands])
        
    reduction = f1 + f2 + f3 - 3
    
    if padding = 'original':
        image_segment_array = np.zeros((len(img_list), image_height_size - reduction, image_width_size - reduction, n_bands))
    else:
        image_segment_array = np.zeros((len(img_list), image_height_size, image_width_size, n_bands))
    blurred_segment_array = np.zeros((len(blurred_list), image_height_size, image_width_size, n_bands))

    
    for index in range(len(img_list)):
        image_segment_array[index] = img_list[index]
        blurred_segment_array[index] = blurred_list[index]
        
    return blurred_segment_array, image_segment_array



def training_data_generation(DATA_DIR, img_height_size, img_width_size, f_1, f_2, f_3, factor, pad):
    """ 
    This function is used to read in files from a folder which contains the images which are to be used for training the 
    SRCNN model, then returns 2 numpy arrays containing the training and target data for all the images in the folder so that
    they can be used for SRCNN model training.
    
    Inputs:
    - DATA_DIR: File path of the folder containing the images to be used as training data for SRCNN model. Images can be
                changed to other formats (default image format is .tif)
    - img_height_size: Height of image segment to be used for SRCNN model training
    - img_width_size: Width of image segment to be used for SRCNN model training
    - f_1: size of kernel to be used for the first convolutional layer
    - f_2: size of kernel to be used for the second convolutional layer
    - f_3: size of kernel to be used for the last convolutional filter
    - factor: The upscaling factor by which the SRCNN model should be trained to upscale
    - pad: String which determines whether the input image is padded during model training so as to maintain the original image 
           size ('same') or to use the clipped version as implemented in the paper ('original')
    
    Outputs:
    - blurred_full_array: 4 - Dimensional numpy array of Gaussian blurred images to serve as training data for SRCNN model
    - img_full_array: 4 - Dimensional numpy array of original input image to serve as target data for training SRCNN model
    
    """
    
    if f_1 % 2 == 0 or f_2 % 2 == 0 or f_3 % 2 == 0:
        raise ValueError('Please input odd numbers for f1, f2 and f3.')
        
    if pad not in ['original', 'same']:
        raise ValueError("Please input either 'original' or 'same' for pad.")
    
    img_files = glob.glob(DATA_DIR + '\\' + 'Train_*.tif')
    
    img_array_list = []
    blurred_array_list = []
    
    for file in range(len(img_files)):
        img = np.transpose(gdal.Open(img_files[file]).ReadAsArray(), [1, 2, 0])
        blurred_img = blurred_image_creation(img, n_factor = factor)
    
        if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
            blurred_array, img_array = image_clip_to_segment(img, blurred_img, img_height_size, img_width_size, mode = 0, 
                                                             n_bands = img.shape[2], f1 = f_1, f2 = f_2, f3 = f_3, 
                                                             padding = pad)
        elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
            blurred_array, img_array = image_clip_to_segment(img, blurred_img, img_height_size, img_width_size, mode = 1, 
                                                             n_bands = img.shape[2], f1 = f_1, f2 = f_2, f3 = f_3, 
                                                             padding = pad)
        elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
            blurred_array, img_array = image_clip_to_segment(img, blurred_img, img_height_size, img_width_size, mode = 2, 
                                                             n_bands = img.shape[2], f1 = f_1, f2 = f_2, f3 = f_3, 
                                                             padding = pad)
        else:
            blurred_array, img_array = image_clip_to_segment(img, blurred_img, img_height_size, img_width_size, mode = 3, 
                                                             n_bands = img.shape[2], f1 = f_1, f2 = f_2, f3 = f_3, 
                                                             padding = pad)
        
        img_array_list.append(img_array)
        blurred_array_list.append(blurred_array)
        
    img_full_array = np.concatenate(img_array_list, axis = 0)
    blurred_full_array = np.concatenate(blurred_array_list, axis = 0)
    
    del img_files ; gc.collect()
    
    return blurred_full_array, img_full_array



def srcnn_model(image_height_size, image_width_size, n_bands, n1 = 64, n2 = 32, f1 = 9, f2 = 1, f3 = 5, l_r = 0.00001, pad = 'same'):
    """ 
    This function creates the SRCNN model which needs to be trained, following the main architecture as described in the 
    paper 'Image Super - Resolution Using Deep Convolutional Networks'.
    
    Inputs:
    - image_height_size: Height of image segment to be used for SRCNN model training
    - image_width_size: Width of image segment to be used for SRCNN model training
    - n_bands: Number of channels contained in the input images
    - n1: Number of filters for the first hidden convolutional layer
    - n2: Number of filters for the second hidden convolutional layer
    - f1: size of kernel to be used for the first convolutional layer
    - f2: size of kernel to be used for the second convolutional layer
    - f3: size of kernel to be used for the last convolutional filter
    - l_r: Learning rate to be used by the Adam optimizer
    - pad: String which determines whether the input image is padded during model training so as to maintain the original image 
           size ('same') or to use the clipped version as implemented in the paper ('valid')
    
    Outputs:
    - model: SRCNN model compiled using the parameters defined in the input, and compiled with the Adam optimizer and 
             mean squared error loss function
    
    """
    
    if pad not in ['valid', 'same']:
        raise ValueError("Please input either 'valid' or 'same' for pad.")
    
    img_input = Input(shape = (image_height_size, image_width_size, n_bands))
    conv1 = Conv2D(n1, (f1, f1), padding = pad, activation = 'relu')(img_input)
    conv2 = Conv2D(n2, (f2, f2), padding = pad, activation = 'relu')(conv1)
    conv3 = Conv2D(n_bands, (f3, f3), padding = pad)(conv2)
    
    model = Model(inputs = img_input, outputs = conv3)
    model.compile(optimizer = Adam(lr = l_r), loss = 'mse', metrics = ['mse'])
    
    return model


