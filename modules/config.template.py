#---------------------------- Predicting Models --------------------------------

# path to the model which is used to predict new data
TRAINED_MODEL = ""

# path to the data to be predicted by the model
# can be directory or file
PREDICT_DATA = ""

# path where predictions for new data should be saved
# nothing is saved if empty
PREDICT_RESULTS = ""

#---------------------------- Training Models ----------------------------------

# path to training data (root-dir)
# the root dir needs subdirs with the labels and the corresponding data inside
# of them
# /root-dir
#    - /label_1
#    - /label_2
TRAINING_DATA = ""

# path where the crated model should be saved to
MODEL_PATH = ""

# path to raw data (root-dir)
# the root dir needs subdirs with the labels and the corresponding data inside
# of them
# /root-dir
#    - /label_1
#    - /label_2
RAW_DATA = ""

# image size in pixels to train the model with
# all images are resized to IMG_SIZExIMG_SIZE
IMG_SIZE = 141

# specifies the amount of trainig epochs for the model
TRAINING_EPOCHS = 30

# sets the percentage of training data which is not used for training but for validation
PERCENT_TEST = 0.20

#---------------------------- Other --------------------------------------------

# path to the model used for contour detection (prepareImage.ipynb)
CONTOUR_MODEL = ""

# !! not used by the main scripts !!
# path to test (verification) data (root-dir)
# the root dir needs subdirs with the labels and the corresponding data inside
# of them
# /root-dir
#    - /label_1
#    - /label_2
TEST_DATA = ""