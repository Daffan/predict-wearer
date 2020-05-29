import numpy as np

def normalization(X):

    X = (X - np.sum(X, axis = 1, keep_dim = True)) / np.std(X, axis = 1, keep_dim = True)

    return X

def train_test_split_0(X, y, test_ratio = 0.1):
    '''
    select half of the split from beginning and half from the end
    '''
    num_frames = X.shape[0]
    test_size = int(num_frames * test_ratio)

    idx_test = np.array(range(test_size)) - int(test_size/2.)
    idx_train = np.array(range(test_size, num_frames)) - int(test_size/2.)

    X_train = X[idx_train, :, :]
    X_test = X[idx_test, :, :]
    y_train = y[idx_train]
    y_test = y[idx_test]

    return X_train, y_train, X_test, y_test

def train_test_split_1(X, y, test_ratio = 0.1):
    '''
    select half of the split from beginning and half from the end
    '''
    num_frames = X.shape[0]
    test_size = int(num_frames * test_ratio)

    idx_1 = np.array(np.where(y == True)).reshape(-1)
    idx_0 = np.array(np.where(y == False)).reshape(-1)
    idx_1_test = [idx_1[:test_size//4], idx_1[-test_size//4: -1]]
    idx_0_test = [idx_0[:test_size//4], idx_0[-test_size//4: -1]]
    idx_test = np.concatenate(idx_1_test + idx_0_test)
    idx_train = np.array([i for i in range(num_frames) if i not in idx_test])

    X_train = X[idx_train, :, :]
    X_test = X[idx_test, :, :]
    y_train = y[idx_train]
    y_test = y[idx_test]

    return X_train, y_train, X_test, y_test
