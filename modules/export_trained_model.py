import os
import tempfile
import tensorflow as tf
import pickle
from datetime import datetime
from modules.zip_dir import zip_dir

def export_trained_model(model, epochs, loss, acc, hist, plot, training_data, model_path):
    
    model_loss = f"{loss:.3f}".replace(".","-")
    model_acc = f"{acc:.3f}".replace(".","-")
    training_shape = list(training_data.get_tf_images().get_shape())
    export_file = f"{model_path}/{datetime.now().strftime('%y%m%d%H%M')}_{model.name}_e{epochs}_{model_acc}_{model_loss}_"+\
    f"{training_shape[0]}_{training_shape[1]}_{training_shape[2]}.zip"
    
    with tempfile.TemporaryDirectory() as tmpdir:

        # save the model
        tf.keras.models.save_model(model, tmpdir)

        # save the labels
        with open(tmpdir+"/assets/labels_dict.pickle", "wb") as file:
            pickle.dump(training_data.get_labels_dict(), file, protocol=pickle.HIGHEST_PROTOCOL)

        # save the TensorShape
        with open(tmpdir+"/assets/tensorshape.pickle", "wb") as file:
            pickle.dump(training_data.get_tf_images().shape, file, protocol=pickle.HIGHEST_PROTOCOL)

        # write model summary
        with open(tmpdir+"/assets/model_summary.txt", "w") as file:
            model.summary(print_fn=lambda txt: file.write(txt + '\n'))

        # save the history
        with open(tmpdir+"/assets/history.pickle", "wb") as file:
            pickle.dump(hist.history, file, protocol=pickle.HIGHEST_PROTOCOL)

        # save the graphic
        plot.savefig(tmpdir+"/assets/fig.png", dpi=100, transparent=False, facecolor="w")
    
        # check if model_path exists
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
    
        # zip the temp files to export path
        zip_dir(export_file, tmpdir)