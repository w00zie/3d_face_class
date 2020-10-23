import matplotlib.pyplot as plt
import numpy as np
import itertools
from models import PoolNet, PoolNetv2, PoolNetv3
import torch

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(18, 18))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def get_model(name: str, device: torch.device, num_classes: int) -> torch.nn.Module:
    if name == "pool":
        return PoolNet(num_classes=num_classes).to(device)
    elif name == "pool2":
        return PoolNetv2(num_classes=num_classes).to(device)
    elif name == "pool3":
        return PoolNetv3(num_classes=num_classes).to(device)
    else:
        print(f'{name} not implemented!')
        exit()
