import os
import tensorflow as tf
import cv2 as cv

def load_predict_data(path, picture_width, picture_height):
    predict_data = []
    loaded_files = []
    loaded_files_names = []
    
    # check if path shows directory or file
    if os.path.isdir(path):
        # path is directory - loop throuh all files
        for file in os.listdir(path):
            current_image = cv.imread(os.path.join(path,file))
            resized_image = cv.resize(current_image, (picture_width, picture_height))
            loaded_files.append(current_image)
            loaded_files_names.append(file)
            predict_data.append(tf.constant(resized_image))
    else:
        # path is file
        current_image = cv.imread(os.path.join(path,file))
        resized_image = cv.resize(current_image, (picture_width, picture_height))
        loaded_files.append(current_image)
        predict_data.append(tf.constant(resized_image))
        
    return tf.stack(predict_data), loaded_files, loaded_files_names 