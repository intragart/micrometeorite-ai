# import libraries
import os
import tensorflow as tf
import cv2 as cv

def load_predict_data(path, picture_width, picture_height):
    """Loads a single file or all files in a directory to be used with an
    already trained model. Returns loaded files (tf.stack and raw) and
    filenames

    Keyword arguments:
    path -- path to image or directory with images
    picture_width -- in pixels for the model
    picture_height -- in pixels for the model
    """

    predict_data = [] # contains the pictures as tf.constants
    loaded_files = [] # contains the pictures (raw)
    loaded_files_names = [] # contains the filenames for the pictures
    
    # check if path shows directory or file
    if os.path.isdir(path):
        # path is directory - loop throuh all files
        for file in os.listdir(path):
            # load the current file, resize it and append the data to the
            # arrays
            current_image = cv.imread(os.path.join(path,file))
            resized_image = cv.resize(current_image, (picture_width, picture_height))
            loaded_files.append(current_image)
            loaded_files_names.append(file)
            predict_data.append(tf.constant(resized_image))
    else:
        # path is file
        # load the current file, resize it and append the data to the
        # arrays
        current_image = cv.imread(os.path.join(path,file))
        resized_image = cv.resize(current_image, (picture_width, picture_height))
        loaded_files.append(current_image)
        predict_data.append(tf.constant(resized_image))
        
    return tf.stack(predict_data), loaded_files, loaded_files_names 