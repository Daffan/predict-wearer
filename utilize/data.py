import librosa
import os
import numpy as np
import pandas as pd

DATA_PATH = "C:\\Users\\zifan\\OneDrive\\Desktop\\Zifan Xu\\Speech Recognition\\predict-wearer\\data\\001_0522.wav"
LALBELS_PATH = "C:\\Users\\zifan\\OneDrive\\Desktop\\Zifan Xu\\Speech Recognition\\predict-wearer\\data\\001_0522.txt"

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

def load_frames(data_path = DATA_PATH, labels_path = LALBELS_PATH, frame_length = 256, sr = 8000):
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

    for i in range(len(X_temp)//frame_length):

        if y_temp[(i+1)*frame_length-1, 0] and not y_temp[(i+1)*frame_length-1, 1]:
            y.append(True)
            X.append(X_temp[i*frame_length:(i+1)*frame_length].reshape(1, frame_length))

        elif y_temp[(i+1)*frame_length-1, 1] and not y_temp[(i+1)*frame_length-1, 0]:
            y.append(False)
            X.append(X_temp[i*frame_length:(i+1)*frame_length].reshape(1, frame_length))

    X = np.concatenate(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = np.array(y)

    return X, y, sr

def get_labels_df(labels_path = LALBELS_PATH):

    labels = pd.read_csv(labels_path, sep='\t', header=None,
                        names=['start', 'end', 'annotation'],
                        dtype=dict(start=float,end=float,annotation=str))

    return labels


if __name__ == '__main__':

    X, y, sr = load_frames()

    print("Load audio data of %d frames with a frame length of %d" %(X.shape[0], X.shape[1]))

    # X, y, sr = load_full_frame()

    # print(X.shape)
