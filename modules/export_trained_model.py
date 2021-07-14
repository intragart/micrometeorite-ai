# import libraries
import os
import tempfile
from keras_tuner.engine import tuner
import tensorflow as tf
import pickle
from datetime import datetime
import contextlib

# own modules
from modules.zip_dir import zip_dir

def export_trained_model(model, epochs, loss, acc, hist, hist_plot, cm_plot,
cm_data, roc_plot, exec_times, training_data, model_path, tuner=None):
    """Exports the trained model and its assets into a zipped file.

    Keyword arguments:
    model -- tf.keras.models object
    epochs -- number of trained epochs for the model
    loss -- val_loss value
    acc -- val_acc value
    hist -- object that contains the history data for each epoch of training
    hist_plot -- pyplot figure that shows history data
    cm_plot -- pyplot figure that shows the confusion matrix
    cm_data -- dictionary that contains the detection metrics
    roc_plot -- pyplot figure that shows the roc curve
    exec_times -- times used for execution
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
    export_file = f"{model_path}/{datetime.now().strftime('%y%m%d%H%M')}_{model.name}_e{epochs}_{model_acc}_{model_loss}_{training_shape[0]}_{training_shape[1]}_{training_shape[2]}.zip"
    
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

        # write val acc and val loss
        with open(tmpdir+"/assets/val_acc_and_val_loss.txt", "w") as file:
            file.write(f"val acc: {model_acc.replace('-', '.')}\n")
            file.write(f"val loss: {model_loss.replace('-', '.')}\n")

        # save the training history
        with open(tmpdir+"/assets/history.pickle", "wb") as file:
            pickle.dump(hist.history, file, protocol=pickle.HIGHEST_PROTOCOL)

        # save the training history diagramm
        hist_plot.savefig(tmpdir+"/assets/history.png", dpi=300, transparent=False,
        facecolor="w", bbox_inches="tight")

        # save the confusion matrix
        cm_plot.savefig(tmpdir+"/assets/confusion_matrix.png", dpi=300, transparent=False,
        facecolor="w", bbox_inches="tight")

        # save the detection metrics
        with open(tmpdir+"/assets/detection_metrics.txt", "w") as file:
            for key in cm_data.keys():
                file.write(f"{key}: {cm_data[key]}\n")

        # save the roc curve
        roc_plot.savefig(tmpdir+"/assets/roc.png", dpi=300, transparent=False,
        facecolor="w", bbox_inches="tight")

        # save the execution times
        with open(tmpdir+"/assets/exec_times.txt", "w") as file:
            file.write(f"Start: {exec_times[0]}\n")
            file.write(f"Stop: {exec_times[1]}\n")
            file.write(f"Elapsed: {exec_times[1] - exec_times[0]}\n")

        # save tuner summary if present
        if tuner is not None:
            with open(tmpdir+"/assets/tuner_summary.txt", "w") as file:
                with contextlib.redirect_stdout(file):
                    tuner.results_summary(100)

        # check if model_path exists and create the path if not already exists
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
    
        # zip the temp directory to export path/zipfile
        zip_dir(export_file, tmpdir)