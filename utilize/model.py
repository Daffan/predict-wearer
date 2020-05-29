from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Conv1D, MaxPool1D, UpSampling1D, Dropout

import numpy as np
import tensorflow as tf

class WearerModel(Model):

    def __init__(self, frame_length = 256, num_channels = 1, p = 0.5, regularization = 0.0005):

        super(WearerModel, self).__init__()

        self.frame_length = frame_length
        self.num_channels = num_channels
        self.dropout = p
        self.regularization = regularization

        ## conv autoencoder
        # encoder conv1
        self.conv1 = Conv1D(100, kernel_size = 1, activation = 'relu', input_shape = (frame_length, num_channels))
        self.maxpool1 = MaxPool1D(pool_size = 2)
        # encoder conv2
        self.conv2 = Conv1D(100, kernel_size = 1, activation = 'relu')
        self.maxpool2 = MaxPool1D(pool_size = 2)
        # encoder conv3
        self.conv3 = Conv1D(100, kernel_size = 1, activation = 'relu')
        self.maxpool3 = MaxPool1D(pool_size = 4)
        # decoder conv1
        self.deconv1 = Conv1D(100, kernel_size = 1, activation = 'relu')
        self.upsampling1 = UpSampling1D(size = 4)
        # decoder conv2
        self.deconv2 = Conv1D(100, kernel_size = 1, activation = 'relu')
        self.upsampling2 = UpSampling1D(size = 2)
        # decoder conv3
        self.deconv3 = Conv1D(100, kernel_size = 1, activation = 'relu')
        self.upsampling3 = UpSampling1D(size = 2)
        # one fully-connected layer
        self.dense0 = Dense(1, activation = 'relu')
        self.conv_dropout = Dropout(p)

        ## three layers of bidirectional lstm
        self.bilstm1 = Bidirectional(LSTM(128, dropout=p, return_sequences=True))
        self.bilstm2 = Bidirectional(LSTM(128, dropout=p, return_sequences=True))
        self.bilstm3 = Bidirectional(LSTM(128, dropout=p, return_sequences=True))

        ## four layers of lstm
        self.lstm1 = LSTM(128, dropout=p, kernel_regularizer=regularizers.l2(regularization), return_sequences=True)
        self.lstm2 = LSTM(128, dropout=p, kernel_regularizer=regularizers.l2(regularization), return_sequences=True)
        self.lstm3 = LSTM(128, dropout=p, kernel_regularizer=regularizers.l2(regularization), return_sequences=True)
        self.lstm4 = LSTM(128, dropout=p, kernel_regularizer=regularizers.l2(regularization), return_sequences=True)

        ## three layers of fully connected layers
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.dense3 = Dense(128, activation='relu')
        self.dense4 = Dense(128, activation='relu')

        self.logit = Dense(1, activation='sigmoid')

    def call(self, x):

        ## conv autoencoder
        x = self.conv_dropout(self.maxpool1(self.conv1(x)))
        x = self.conv_dropout(self.maxpool2(self.conv2(x)))
        x = self.conv_dropout(self.maxpool3(self.conv3(x)))
        x = self.conv_dropout(self.upsampling1(self.deconv1(x)))
        x = self.conv_dropout(self.upsampling2(self.deconv2(x)))
        x = self.conv_dropout(self.upsampling3(self.deconv3(x)))
        # x = self.dense0(x)
        ## three layers bidirectional lstm

        x = self.bilstm1(x)
        x = self.bilstm2(x)
        x = self.bilstm3(x)
        ## four layers lstm
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.lstm4(x)
        ## four layers lstm
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        ## logit
        x = self.logit(x)
        x = x[:, -1, :]

        return x

class ConvLSTM(Model):

    def __init__(self, frame_length = 256, num_channels = 1, p = 0.2):

        super(ConvLSTM, self).__init__()

        self.conv1 = Conv1D(64, kernel_size = 5, activation = 'relu', input_shape = (frame_length, num_channels))
        self.conv2 = Conv1D(64, kernel_size = 5, activation = 'relu')
        self.conv3 = Conv1D(64, kernel_size = 5, activation = 'relu')
        self.conv4 = Conv1D(64, kernel_size = 5, activation = 'relu')

        self.lstm1 = LSTM(128, return_sequences=True)
        self.lstm2 = LSTM(128, return_sequences=True)

        self.dense = Dense(1, activation = 'sigmoid')

        self.dropout = Dropout(p)

    def call(self, x):

        x = self.dropout(self.conv1(x))
        x = self.dropout(self.conv2(x))
        x = self.dropout(self.conv3(x))
        x = self.dropout(self.conv4(x))

        x = self.dropout(self.lstm1(x))
        x = self.dropout(self.lstm2(x))

        x = self.dense(x)
        x = x[:, -1, :]

        return x

class BaselineCNN(Model):

    def __init__(self, frame_length = 256, num_channels = 1):

        super(BaselineCNN, self).__init__()

        self.conv1 = Conv1D(64, kernel_size = 5, activation = 'relu', input_shape = (frame_length, num_channels))
        self.conv2 = Conv1D(64, kernel_size = 5, activation = 'relu')
        self.conv3 = Conv1D(64, kernel_size = 5, activation = 'relu')
        self.conv4 = Conv1D(64, kernel_size = 5, activation = 'relu')

        self.dense1 = Dense(128, activation = 'relu')
        self.dense2 = Dense(128, activation = 'relu')

        self.dense = Dense(1, activation = 'sigmoid')

    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.dense1(x)
        x = self.dense2(x)

        x = self.dense(x)
        x = x[:, -1, :]

        return x

if __name__ == '__main__':

    model = WearerModel()
    model = ConvLSTM()
    model = BaselineCNN()
