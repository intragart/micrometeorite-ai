import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def roc(model, val_images, val_labels):
    #ROC Kurve
    # calculate the fpr and tpr for all thresholds of the classification
    y_val_argmax = np.argmax(val_labels, axis = 1)
    probs = model.predict(val_images)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_val_argmax, preds, pos_label = 1)
    roc_auc = metrics.auc(fpr, tpr)

    print(len(y_val_argmax))

    # method I: plt
    """
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    """

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_title('Receiver Operating Characteristic')
    ax1.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax1.legend(loc = 'lower right')
    ax1.plot([0, 1], [0, 1],'r--')
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.05])
    ax1.set_ylabel('True Positive Rate')
    ax1.set_xlabel('False Positive Rate')
    
    return fig