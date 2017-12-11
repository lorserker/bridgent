import numpy as np

def acc012(labels, predictions):
    n = labels.shape[0]
    error = np.abs(labels - np.round(predictions))
    perfect = np.sum(error == 0) / n
    off_1 = np.sum(error <= 1) / n
    off_2 = np.sum(error <= 2) / n
    return (perfect, off_1, off_2)
    