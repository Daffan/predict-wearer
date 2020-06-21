import numpy as np

def normalization(X):

    X = (X - np.sum(X, axis = 1, keep_dim = True)) / np.std(X, axis = 1, keep_dim = True)

    return X

def train_test_split_unbalanced_class(X, y, test_ratio = 0.1):
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

def train_test_split_balanced_class(X, y, test_ratio = 0.1):
    '''
    select half of the split from beginning and half from the end. The test
    set is a continous segment cropped half from beginning and half from the
    end, and the num of frames are the same for both of the class
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

def balance_multi_classes(X, y, seed = 42):
    '''
    Down sample the data with three classes. Return equal-sized continous segments
    from each of the class. The segments are cropped half from the beginning and
    half from the end.
    '''

    np.random.seed(seed)

    idx_0, = np.where((y == np.array([True, False, False])).all(axis = 1))
    idx_1, = np.where((y == np.array([False, True, False])).all(axis = 1))
    idx_2, = np.where((y == np.array([False, False, True])).all(axis = 1))

    min_length = min(len(idx_0), len(idx_1), len(idx_2))

    idx_0 = np.concatenate([idx_0[:min_length//2], idx_0[-min_length//2:]])
    idx_1 = np.concatenate([idx_1[:min_length//2], idx_1[-min_length//2:]])
    idx_2 = np.concatenate([idx_2[:min_length//2], idx_2[-min_length//2:]])

    idx = np.concatenate([idx_0, idx_1, idx_2])

    return X[idx], y[idx]

def train_test_split_multi_classes(X, y, p = 0.1, seed = 42):
    '''
    select half of the split from beginning and half from the end. The test
    set is a continous segment cropped half from beginning and half from the
    end, and the num of frames are the same for three classes
    '''

    np.random.seed(seed)

    idx_0, = np.where((y == np.array([True, False, False])).all(axis = 1))
    idx_1, = np.where((y == np.array([False, True, False])).all(axis = 1))
    idx_2, = np.where((y == np.array([False, False, True])).all(axis = 1))

    min_length = min(len(idx_0), len(idx_1), len(idx_2))

    idx_test = np.concatenate([idx_0[:np.int(min_length*p/2)], idx_1[:np.int(min_length*p/2)], idx_2[:np.int(min_length*p/2)],
                               idx_0[-np.int(min_length*p/2):], idx_1[-np.int(min_length*p/2):], idx_2[-np.int(min_length*p/2):]])
    idx_train = np.concatenate([idx_0[np.int(min_length*p/2):-np.int(min_length*p/2)], idx_1[np.int(min_length*p/2):-np.int(min_length*p/2)], idx_2[np.int(min_length*p/2):-np.int(min_length*p/2)]])

    return X[idx_train], y[idx_train], X[idx_test], y[idx_test]
