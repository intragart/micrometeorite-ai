# import libraries
import tensorflow as tf
from zipfile import ZipFile
import pickle
import tempfile

def load_model(model_file):
    """Loads the exported model from the zip-file. Returns the tf.keras.model-
    object, the dictionary which contains the trained labels for the model and
    the tensorshape of the training data

    Keyword arguments:
    model_file -- path to the zip filewhich contains the trained model
    """
    
    # we use a temporary directory to extract the zipfile
    with tempfile.TemporaryDirectory() as tmpdir:

        # extracts the zipfile
        with ZipFile(model_file, "r") as zf:
            zf.extractall(tmpdir)
            
        # load the labels dictionary
        with open(f"{tmpdir}/assets/labels_dict.pickle", "rb") as file:
            labels = pickle.load(file)

        # load tensorshape
        with open(f"{tmpdir}/assets/tensorshape.pickle", "rb") as file:
            tensorshape = pickle.load(file)
            
        # load model itself
        model = tf.keras.models.load_model(tmpdir)
        
        # Returns the tf.keras.model- object, the dictionary which contains the
        # trained labels for the model and the tensorshape of the training data
        return model, labels, tensorshape