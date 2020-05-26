import numpy as np

def normalization(X):

    X = (X - np.sum(X, axis = 1, keep_dim = True)) / np.std(X, axis = 1, keep_dim = True)

    return X
