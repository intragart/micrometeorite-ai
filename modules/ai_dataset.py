# hallo
import os # to interact with the system
import tensorflow as tf # for modell training
import cv2 as cv
import random
from tqdm import tqdm, trange

class ai_dataset:
    __labels_dict = {}
    
    def __init__(self, source_path="", img_size=0, tf_ready=False):
        # define attributes for instance
        self.__images = []
        self.__labels = []
        self.__tf_images = tf.constant([])
        self.__tf_labels = tf.constant([])
        self.__img_size = 0 # 0 means not defined - images can have different sizes
        self.__tf_ready = False # indicates if __images and __labels have been filled
        
        if len(source_path):
            self.load_data(source_path)
            
        if img_size > 0:
            self.resize(img_size)
            
        if tf_ready == True:
            self.shuffle()
            self.tf_ready()
        
    def load_data(self, source_path):
        # local variables
        arr_images = []
        arr_labels = []
        labels = os.listdir(source_path)
        
        # loop through each label in source_path
        for label in labels:
            path = os.path.join(source_path, label)
            file_list = os.listdir(path)
            label_number = 0
            exception_counter = 0
            error_counter = 0
            
            # add label to dict or get label_number for label
            if label in ai_dataset.__labels_dict:
                # label already exists, get label number
                label_number = ai_dataset.__labels_dict.get(label)
            else:
                # new label, add to dict and set label number
                label_number = len(ai_dataset.__labels_dict)
                ai_dataset.__labels_dict[label] = label_number
            
            # loop through all files inside the labeled folder
            for i in trange(len(file_list)):
                try:
                    # try to read the image
                    current_image = cv.imread(os.path.join(path,file_list[i]))
                    
                    # make sure cv.imread didn't return NULL
                    if current_image is not None:
                        self.__images.append(current_image)
                        self.__labels.append(label_number)
                    else:
                        error_counter += 1
                except Exception as e:
                    # an exception occured
                    exception_counter += 1

            print(f"Errors: {error_counter}; Exceptions: {exception_counter}")
            self.__tf_ready = False
            
    def shuffle(self):
        raw_dataset = []
        
        for i in range(len(self.__images)):
            raw_dataset.append([self.__images[i], self.__labels[i]])
            
        random.shuffle(raw_dataset)
        
        self.__images = []
        self.__labels = []
        
        for i in range(len(raw_dataset)):
            self.__images.append(raw_dataset[i][0])
            self.__labels.append(raw_dataset[i][1])
            
        self.__tf_ready = False
            
    def resize(self, new_size):
        for i in range(len(self.__images)):
            current_image = self.__images[i]
            self.__images[i] = cv.resize(current_image, (new_size, new_size))
            
        self.__img_size = new_size
        self.__tf_ready = False
        
    def tf_ready(self):
        arr_images = []
        arr_labels = []
        
        for i in range(len(self.__images)):
            arr_images.append(tf.constant(self.__images[i]))
            arr_labels.append(tf.constant(self.__labels[i]))
        
        self.__tf_images = tf.stack(arr_images)
        self.__tf_labels = tf.stack(arr_labels)
        
        self.__tf_ready = True
        print(f"{len(self.__images)} Images ready")
        
    def get_tf_images(self):
        if self.__tf_ready == True:
            return self.__tf_images
        else:
            self.tf_ready()
            return self.__tf_images
    
    def get_tf_labels(self):
        if self.__tf_ready == True:
            return self.__tf_labels
        else:
            self.tf_ready()
            return self.__tf_labels
        
    def get_image(self, i):
        return self.__images[i]
    
    def get_label(self, i):
        return list(ai_dataset.__labels_dict)[self.__labels[i]]
    
    def get_status(self):
        print(f"Image Data: {self.get_tf_images().shape}")
        print(f"Label Data: {self.get_tf_labels().shape}")
        print(f"Class labels: {ai_dataset.__labels_dict.items()}")
        
    def get_labels_count(self):
        return len(ai_dataset.__labels_dict)
    
    def get_labels_dict(self):
        return ai_dataset.__labels_dict