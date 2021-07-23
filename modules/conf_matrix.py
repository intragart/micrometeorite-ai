import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def conf_matrix(model, val_images, val_labels):
    
    predictions = model.predict(val_images)
    #print(predictions[:50])

    predictions = np.argmax(predictions, axis = 1)
    #print(predictions[:6])
    predictions = pd.get_dummies(predictions).values
    #print(predictions[:6])

    df_predictions = pd.DataFrame(predictions, columns = ['Nicht-Mikrometeorit' , 'Mikrometeorit'])
    df_predictions['KategoriePred'] = df_predictions.idxmax(1)
    df_predictions = df_predictions.drop('Mikrometeorit', 1)
    df_predictions = df_predictions.drop('Nicht-Mikrometeorit', 1)
    #df_predictions

    df_Y_val = pd.DataFrame(val_labels, columns = ['Nicht-Mikrometeorit' , 'Mikrometeorit'])
    df_Y_val['KategorieYval'] = df_Y_val.idxmax(1)
    df_Y_val = df_Y_val.drop('Mikrometeorit', 1)
    df_Y_val = df_Y_val.drop('Nicht-Mikrometeorit', 1)
    #df_Y_val

    confusionMatrix = confusion_matrix(df_Y_val, df_predictions, labels=["Nicht-Mikrometeorit", "Mikrometeorit"])
    df_confusion = pd.DataFrame(confusionMatrix)
    #df_confusion
    #confusionMatrix

    conf_matrix = plot_confusion_matrix(cm = confusionMatrix, 
                      normalize    = False,
                      target_names = ['Nicht-Mikrometeorit', 'Mikrometeorit'],
                      title        = "Confusion Matrix")

    #Erkennungsmetriken
    #print(confusionMatrix)

    tp = confusionMatrix[1][1]
    fp = confusionMatrix[0][1]
    fn = confusionMatrix[1][0]
    tn = confusionMatrix[0][0]

    #True-Positive Rate, gleichbedeutend mit Recall
    tpr_man = tp / (tp + fn)

    #False-Positive Rate
    fpr_man = fp / (fp + tn)

    #Precision
    precision = tp / (tp + fp)

    erkennungsmetriken = {"TPR" : tpr_man, "FPR" : fpr_man, "Precision" : precision}
                
    return conf_matrix, erkennungsmetriken

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion Matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot()
    im = ax1.imshow(cm, interpolation='nearest', cmap=cmap)
    ax1.set_title(title)
    fig.colorbar(im, ax=ax1)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        #ax1.set_xticks(tick_marks, target_names)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(target_names, rotation=45)
        #ax1.set_yticks(tick_marks, target_names)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax1.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            ax1.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    fig.tight_layout()
    ax1.set_ylabel('True label')
    ax1.set_xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    return fig