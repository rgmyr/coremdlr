"""
Model diagrams and perfomance plots, etc.
"""
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



def keras_network_diagram(model, to_file=None, **kwargs):
    """
    Save pydot graph of network, or return it as SVG object.
    """
    if to_file:
        keras_utils.plot_model(model, to_file=to_file, **kwargs)
    else:
        dot_graph = keras_utils.vis_utils.model_to_dot(model, **kwargs)
        return SVG(dot_graph.create(prog='dot', format='svg'))


def confusion_matrix_plot(cm, classes,
                          normalize=False,
                          title=None, ax=None,
                          figsize=(10,10),
                          cmap=plt.cm.Blues):
    """
    Plot a confusion matrix, return matplotlib Axes.

    Parameters
    ----------
    cm : array, dict, or DataFrame
        Confusion matrix, or data structure containing y_true & y_pred.
        If dict or DataFrame, must have 'y_true' and 'y_pred' keys.
        If column array, should contain y_true in first column, y_pred in second.
        Otherwise, should be square matrix with all dims == len(classes).
    classes : list(str)
        List of the names of classes corresponding to each label.
    normalize : bool, optional
        Whether to row normalize the cm before plotting, default=False.
    """
    if type(cm) is dict:
        cm = confusion_matrix(cm['y_true'], cm['y_pred'])
    elif type(cm) is pd.core.frame.DataFrame:
        cm = confusion_matrix(cm['y_true'].values, cm['y_pred'].values)
    elif cm.shape[0] != 2 and cm.shape[1] == 2:
        cm = confusion_matrix(cm[:,0], cm[:,1])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if title is None:
            title = "Normalized confusion matrix"
    elif title is None:
        title = 'Raw confusion matrix'

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.grid(False)
    #plt.colorbar(im, ax=ax)

    # Plot Labels
    ax.set_title(title)

    tick_marks = np.arange(len(classes))

    ax.set_ylabel('True label')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    ax.set_xlabel('Predicted label')
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.tick_params(axis='x', labelrotation=45)

    # CM cell values
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    return ax
