import librosa
import os
import numpy as np
import pandas as pd

DATA_PATH = "C:\\Users\\zifan\\OneDrive\\Desktop\\Zifan Xu\\Speech Recognition\\predict-wearer\\data\\002_0528.wav"
LALBELS_PATH = "C:\\Users\\zifan\\OneDrive\\Desktop\\Zifan Xu\\Speech Recognition\\predict-wearer\\data\\002_0528.txt"

def load_full_frame(data_path = DATA_PATH, labels_path = LALBELS_PATH, sr = 8000):
    '''
    Load the data in 1D real-time sequence with labels for each of the
    time step and specified sample rate
    keyword arguments:
        data_path [str] -- path to where the audio data (.wav) locate
        label_path [str] -- path to where the labels file (.txt) local
        sr [int] -- sample rate to load the audio
    returns:
        X [ndarray] -- audio data in time sequence
        y [ndarray] -- labels data
        sr [int] -- sample rate
    '''

    if not os.path.exists(data_path):
        raise RuntimeError("Data not found in the specified path: %s" %DATA_PATH)

    if not os.path.exists(labels_path):
        raise RuntimeError("Labels not found in the specified path: %s" %LALBELS_PATH)

    X, sr = librosa.load(data_path, sr = sr)

    # Labels appears in start time and end time and the annotation associate
    # with all the data in the period of time
    labels = pd.read_csv(labels_path, sep='\t', header=None,
                        names=['start', 'end', 'annotation'],
                        dtype=dict(start=float,end=float,annotation=str))

    # Create labels in shape (num_instances, num_speakers(2))
    # Two speakers specified here: wearer and others
    # [0, 1] for time step t means wearer is not speaking while others is
    y = np.zeros((len(X), 2), dtype = np.bool)
    for index in labels.index:
        start, end, label = labels.loc[index, 'start'], \
                            labels.loc[index, 'end'], \
                            labels.loc[index, 'annotation']
        start_idx = int(start*sr)
        end_idx = int(end*sr)
        if label == 'W':
            y[start_idx:end_idx, 0] = True
        elif label == 'O':
            y[start_idx:end_idx, 1] = True

    return X, y, sr

def load_frames(data_path = DATA_PATH, labels_path = LALBELS_PATH, frame_length = 512, hop_length = 256, sr = 8000):
    '''
    Load the data in sliding frames with specified frame_length. The label
    of a frame is decided by the label of the last time step of a frame.
    Keyword arguments:
        frame_length [int] -- number of time step in a frame
    Returns:
        X [ndarray] -- frames in shape (num_frames, frame_length, 1)
        y [ndarray] -- labels i nshape (num_frame,)
    '''
    X_temp, y_temp , sr = load_full_frame(data_path, labels_path, sr)
    X, y = [], []
    i = 0
    while i * hop_length + frame_length <= X_temp.shape[0]:

        if y_temp[i * hop_length + frame_length - 1, 0] and not y_temp[i * hop_length + frame_length - 1, 1]:
            y.append(True)
            X.append(X_temp[(i*hop_length):(i*hop_length+frame_length)].reshape(1, frame_length))

        elif y_temp[i * hop_length + frame_length - 1, 1] and not y_temp[i * hop_length + frame_length - 1, 0]:
            y.append(False)
            X.append(X_temp[(i*hop_length):(i*hop_length+frame_length)].reshape(1, frame_length))
        i += 1

    X = np.concatenate(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = np.array(y)

    return X, y, sr

def get_labels_df(labels_path = LALBELS_PATH):

    labels = pd.read_csv(labels_path, sep='\t', header=None,
                        names=['start', 'end', 'annotation'],
                        dtype=dict(start=float,end=float,annotation=str))

    return labels

def load_mfcc_frames(data_path = DATA_PATH, labels_path = LALBELS_PATH,
                    frame_length = 960, hop_length = 160, sr = 16000, M = 12):
    '''
    Load the mfcc coefficeints and their first and second difference as features vectors
    using sliding windows of frame_length (960 by default) and hop_length (160).
    M=12 frames forms one single time context
    Return:
        X [ndarray] -- time context windows in shape (num_time_context, M, 42)
        y [ndarray] -- label for each time context window (num_time_context,)
                        1 (0) represent wearer (other) speech.
    '''
    X_temp, y_temp , sr = load_full_frame(data_path, labels_path, sr)
    mfcc_14 = librosa.feature.mfcc(X_temp, sr = sr, n_mfcc = 14, n_fft = frame_length,
                                    hop_length = hop_length, center = False)
    mfcc_delta_1 = librosa.feature.delta(mfcc_14)
    mfcc_delta_2 = librosa.feature.delta(mfcc_delta_1)
    mfcc = np.concatenate([mfcc_14, mfcc_delta_1, mfcc_delta_2])

    y_prime = []
    for label in y_temp:
        if label[0] and not label[1]:
            y_prime.append(1)
        elif label[1] and not label[0]:
            y_prime.append(-1)
        else:
            y_prime.append(0)

    y_prime = np.array(y_prime)
    y_frames = librosa.util.frame(y_prime, frame_length = 960, hop_length = 160)
    y_frames = np.round(np.sum(y_frames, axis = 0)/np.float(frame_length)).astype(np.int)

    assert y_frames.shape[0] == mfcc.shape[1] # have the same number of frames

    X, y = [], []
    i = 0

    while True:
        # if all the frames from i to i+M are not ambient sound
        if (y_frames[i: i+M] != 0).all():
            # label 1 if M frames have more wearer speech's label
            # label 0 if M frames have more other speech's label
            y.append((np.sum(y_frames[i: i+M])/np.float(M)).astype(np.int) == 1)
            X.append(np.expand_dims(np.expand_dims(mfcc[:, i: i+M], axis = 0), axis = -1))
            i = i + M # Move to next time context
        else:
            i = i + 1
        if i + M > y_frames.shape[0]:
            break

    X = np.concatenate(X)
    y = np.array(y)

    return X, y, sr

def load_mfcc_frames_multiclass(data_path = DATA_PATH, labels_path = LALBELS_PATH,
                    frame_length = 960, hop_length = 160, sr = 16000, M = 12):
    '''
    Load the 42 mfcc features time context window with multiclass labels
    Return:
    X [ndarray] -- time context windows in shape (num_time_context, M, 42)
    y [ndarray] -- bool matrix in shape (num_frame, 3) with three columns
                    representing wearer's speech, non-wearer and ambient sound repectively.
    '''
    X_temp, y_temp , sr = load_full_frame(data_path, labels_path, sr)
    mfcc_14 = librosa.feature.mfcc(X_temp, sr = sr, n_mfcc = 14, n_fft = frame_length,
                                    hop_length = hop_length, center = False)
    mfcc_delta_1 = librosa.feature.delta(mfcc_14)
    mfcc_delta_2 = librosa.feature.delta(mfcc_delta_1)
    mfcc = np.concatenate([mfcc_14, mfcc_delta_1, mfcc_delta_2])

    y_prime = []
    for label in y_temp:
        if label[0] and not label[1]:
            y_prime.append(1)
        elif label[1] and not label[0]:
            y_prime.append(-1)
        else:
            y_prime.append(0)

    y_prime = np.array(y_prime)
    y_frames = librosa.util.frame(y_prime, frame_length = 960, hop_length = 160)
    y_frames = np.round(np.sum(y_frames, axis = 0)/np.float(frame_length)).astype(np.int)

    assert y_frames.shape[0] == mfcc.shape[1] # have the same number of frames

    X, y = [], []
    i = 0

    while True:
        # wearer's speech
        if (y_frames[i: i+M] == 1).all():
            y.append([True, False, False])
            X.append(np.expand_dims(np.expand_dims(mfcc[:, i: i+M], axis = 0), axis = -1))
            i = i + M
        # other's speech
        elif (y_frames[i: i+M] == -1).all():
            y.append([False, True, False])
            X.append(np.expand_dims(np.expand_dims(mfcc[:, i: i+M], axis = 0), axis = -1))
            i = i + M
        # ambient sound
        elif (y_frames[i: i+M] == 0).all():
            y.append([False, False, True])
            X.append(np.expand_dims(np.expand_dims(mfcc[:, i: i+M], axis = 0), axis = -1))
            i = i + M
        else:
            i = i + 1
        if i + M > y_frames.shape[0]:
            break

    X = np.concatenate(X)
    y = np.array(y)

    return X, y, sr

def load_frames_three_class(data_path = DATA_PATH, labels_path = LALBELS_PATH, \
                            frame_length = 512, hop_length = 256, sr = 8000):
    '''
    Load the data in sliding frames with specified frame_length. The label
    of a frame is decided by the label of the last time step of a frame.
    Keyword arguments:
        frame_length [int] -- number of time step in a frame
    Returns:
        X [ndarray] -- frames in shape (num_frames, frame_length, 1)
        y [ndarray] -- bool matrix in shape (num_frame, 3) with three columns
                       representing wearer's speech, non-wearer and ambient sound repectively.
    '''
    X_temp, y_temp , sr = load_full_frame(data_path, labels_path, sr)
    X, y = [], []
    i = 0
    while i * hop_length + frame_length <= X_temp.shape[0]:
        # wearer speech
        if y_temp[i * hop_length + frame_length - 1, 0] and not y_temp[i * hop_length + frame_length - 1, 1]:
            y.append([True, False, False])
            X.append(X_temp[(i*hop_length):(i*hop_length+frame_length)].reshape(1, frame_length))
        # Non-wearer speech
        elif y_temp[i * hop_length + frame_length - 1, 1] and not y_temp[i * hop_length + frame_length - 1, 0]:
            y.append([False, True, False])
            X.append(X_temp[(i*hop_length):(i*hop_length+frame_length)].reshape(1, frame_length))
        # Ambient sound
        elif not y_temp[i * hop_length + frame_length - 1, 1] and not y_temp[i * hop_length + frame_length - 1, 0]:
            y.append([False, False, True])
            X.append(X_temp[(i*hop_length):(i*hop_length+frame_length)].reshape(1, frame_length))

        i += 1

    X = np.concatenate(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = np.array(y)

    return X, y, sr

if __name__ == '__main__':

    X, y, sr = load_frames()
    # print(X.shape, y.shape, sr)
    print("Load audio data of %d frames with a frame length of %d" %(X.shape[0], X.shape[1]))

    # X, y, sr = load_full_frame()

    # print(X.shape)
