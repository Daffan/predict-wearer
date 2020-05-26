from utilize.data import *
from utilize.model import *
import tensorflow.keras as keras

LOG_FOLDER = 'C:\\Users\\zifan\\OneDrive\\Desktop\\Zifan Xu\\Speech Recognition\\predict-wearer\\results\\001_0524'

model = ConvLSTM()
model.compile(optimizer = keras.optimizers.RMSprop(learning_rate=0.001, decay = 0.01/50),
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])

X, y, sr = load_frames(sr = 8000)
X_normal = (X - np.mean(X, axis = 1, keepdims = True)) / np.std(X, axis = 1, keepdims = True)

callback = tf.keras.callbacks.TensorBoard(log_dir = LOG_FOLDER, histogram_freq=1)

model.fit(X_normal, y, epochs=50, verbose = 0, callbacks = [callback], validation_split = 0.1)
