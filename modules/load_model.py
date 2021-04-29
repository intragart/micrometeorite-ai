import tensorflow as tf
from zipfile import ZipFile
import pickle
import tempfile

def load_model(model_file):
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile(model_file, "r") as zf:
            zf.extractall(tmpdir)
            
        # load the labels
        with open(f"{tmpdir}/assets/labels_dict.pickle", "rb") as file:
            labels = pickle.load(file)

        # load tensor shape
        with open(f"{tmpdir}/assets/tensorshape.pickle", "rb") as file:
            tensorshape = pickle.load(file)
            
        # load model
        model = tf.keras.models.load_model(tmpdir)
            
        return model, labels, tensorshape