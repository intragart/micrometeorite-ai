# import libraries
import os
import tempfile
import tensorflow as tf
import pickle
from datetime import datetime

# own modules
from modules.zip_dir import zip_dir

def export_trained_model(model, epochs, loss, acc, hist, plot, training_data,
model_path):
    """Exports the trained model and its assets into a zipped file.

    Keyword arguments:
    model -- tf.keras.models object
    epochs -- number of trained epochs for the model
    loss -- val_loss value
    acc -- val_acc value
    hist -- object that contains the history data for each epoch of training
    plot -- pyplot figure that shows history data
    training_data -- ai_dataset-object with the training data (not exported!)
    model_path -- path where the zip-file should be saved
    """
    
    # get loss and accuracy values and replace the dot with a minus for the
    # filename of the exported model
    model_loss = f"{loss:.3f}".replace(".","-")
    model_acc = f"{acc:.3f}".replace(".","-")

    # write the tensorshape as list (Number of Images, Pixelwidth, Pixelheight,
    # Pixeldepth)
    training_shape = list(training_data.get_tf_images().get_shape())

    # build the export-path with filename
    export_file = f"{model_path}/{datetime.now().strftime('%y%m%d%H%M')}_\
        {model.name}_e{epochs}_{model_acc}_{model_loss}_{training_shape[0]}_\
            {training_shape[1]}_{training_shape[2]}.zip"
    
    # we use a temporary directory to create the structure for our zip file
    # once everything is exported the entire directory is zipped and deleted
    # afterwards
    with tempfile.TemporaryDirectory() as tmpdir:

        # save the keras-model
        tf.keras.models.save_model(model, tmpdir)

        # save the trained labels as dictionary
        with open(tmpdir+"/assets/labels_dict.pickle", "wb") as file:
            pickle.dump(training_data.get_labels_dict(), file,
            protocol=pickle.HIGHEST_PROTOCOL)

        # save the TensorShape of the training data
        with open(tmpdir+"/assets/tensorshape.pickle", "wb") as file:
            pickle.dump(training_data.get_tf_images().shape, file,
            protocol=pickle.HIGHEST_PROTOCOL)

        # write model summary for the end user
        with open(tmpdir+"/assets/model_summary.txt", "w") as file:
            model.summary(print_fn=lambda txt: file.write(txt + '\n'))

        # save the training history
        with open(tmpdir+"/assets/history.pickle", "wb") as file:
            pickle.dump(hist.history, file, protocol=pickle.HIGHEST_PROTOCOL)

        # save the training history diagramm
        plot.savefig(tmpdir+"/assets/fig.png", dpi=100, transparent=False,
        facecolor="w")
    
        # check if model_path exists and create the path if not already exists
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
    
        # zip the temp directory to export path/zipfile
        zip_dir(export_file, tmpdir)