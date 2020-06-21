from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Conv1D, MaxPool1D, UpSampling1D, Dropout, Conv2D, MaxPool2D, Flatten

import numpy as np
import tensorflow as tf

class VGG_Slimmer(Model):

    def __init__(self, M = 12, feature_vector = 42, p = 0.4, num_output = 3):
'''
num_output > 1, model will output softmax for multiclass classification
'''
        super(VGG_Slimmer, self).__init__()

        self.M = M
        self.feature_vector = feature_vector
        self.p = p


        self.conv1 = Conv2D(64, kernel_size = (3, 3), strides = (1, 1), input_shape = (feature_vector, M, 1))
        self.maxpool1 = MaxPool2D(pool_size = (2, 1), strides = (2, 1))

        self.conv2 = Conv2D(128, kernel_size = (3, 3), strides = (1, 1))
        self.maxpool2 = MaxPool2D(pool_size = (2, 1), strides = (2, 1))

        self.conv3 = Conv2D(256, kernel_size = (3, 3), strides = (1, 1))
        self.maxpool3 = MaxPool2D(pool_size = (2, 2), strides = (2, 2))

        self.conv4 = Conv2D(512, kernel_size = (3, 3), strides = (1, 1))

        self.dense1 = Dense(1024, activation = 'relu')
        self.dense2 = Dense(128, activation = 'relu')

        if num_output > 1:
            self.logit = Dense(num_output, activation = 'softmax')
        elif num_output == 1:
            self.logit = Dense(num_output, activation = 'sigmoid')

        self.dropout = Dropout(p)

    def call(self, x):

        x = self.dropout(self.maxpool1(self.conv1(x)))
        x = self.dropout(self.maxpool2(self.conv2(x)))
        x = self.dropout(self.maxpool3(self.conv3(x)))
        x = self.dropout((self.conv4(x)))

        x = Flatten()(x)

        x = self.dropout(self.dense1(x))
        x = self.dropout(self.dense2(x))

        x = self.logit(x)

        return x


class WearerModel(Model):

    def __init__(self, frame_length = 256, num_channels = 1, p = 0.5, regularization = 0.0005, num_output = 3):
'''
num_output > 1, model will output softmax for multiclass classification
'''
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

        if num_output > 1:
            self.logit = Dense(num_output, activation = 'softmax')
        elif num_output == 1:
            self.logit = Dense(num_output, activation = 'sigmoid')

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

    def __init__(self, frame_length = 256, num_channels = 1, p = 0.2, num_output = 1):
'''
num_output > 1, model will output softmax for multiclass classification
'''
        super(ConvLSTM, self).__init__()

        self.frame_length = frame_length
        self.num_channels = 1
        self.p = p

        self.conv1 = Conv1D(64, kernel_size = 5, activation = 'relu', input_shape = (frame_length, num_channels))
        self.conv2 = Conv1D(64, kernel_size = 5, activation = 'relu')
        self.conv3 = Conv1D(64, kernel_size = 5, activation = 'relu')
        self.conv4 = Conv1D(64, kernel_size = 5, activation = 'relu')

        self.lstm1 = LSTM(128, return_sequences=True)
        self.lstm2 = LSTM(128, return_sequences=True)

        if num_output > 1:
            self.logit = Dense(num_output, activation = 'softmax')
        elif num_output == 1:
            self.logit = Dense(num_output, activation = 'sigmoid')

        self.dropout = Dropout(p)

    def call(self, x):

        x = self.dropout(self.conv1(x))
        x = self.dropout(self.conv2(x))
        x = self.dropout(self.conv3(x))
        x = self.dropout(self.conv4(x))

        x = self.dropout(self.lstm1(x))
        x = self.dropout(self.lstm2(x))

        x = self.logit(x)
        x = x[:, -1, :]

        return x

class BaselineCNN(Model):

    def __init__(self, frame_length = 256, num_channels = 1, num_output = 1):

        super(BaselineCNN, self).__init__()

        self.conv1 = Conv1D(64, kernel_size = 5, activation = 'relu', input_shape = (frame_length, num_channels))
        self.conv2 = Conv1D(64, kernel_size = 5, activation = 'relu')
        self.conv3 = Conv1D(64, kernel_size = 5, activation = 'relu')
        self.conv4 = Conv1D(64, kernel_size = 5, activation = 'relu')

        self.dense1 = Dense(128, activation = 'relu')
        self.dense2 = Dense(128, activation = 'relu')

        if num_output > 1:
            self.logit = Dense(num_output, activation = 'softmax')
        elif num_output == 1:
            self.logit = Dense(num_output, activation = 'sigmoid')

    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.dense1(x)
        x = self.dense2(x)

        x = self.logit(x)
        x = x[:, -1, :]

        return x


if __name__ == '__main__':

    # model = WearerModel()
    # model = ConvLSTM()
    # model = BaselineCNN()
    model = VGG_Slimmer()
    print(model(np.ones((1, 42, 12, 1))).shape)
