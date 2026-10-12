from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D

def build_model():
    model = Sequential([
        Conv1D(32, kernel_size=16, strides=2, activation='relu', padding='same', input_shape=(800, 1), name='conv1'),
        Conv1D(64, 8, strides=1, activation='relu', padding='same', name='conv2'),
        MaxPooling1D(pool_size=2, name='maxpool'),
        Dropout(0.25 , name='dropout1'),
        Flatten(name='flatten'),
        Dense(128, activation='relu', name='dense1'),
        Dropout(0.5, name='dropout2'),
        Dense(1, activation='linear')
    ])
    return model

def layer_name():
    return ['conv1', 'conv2', 'maxpool', 'dropout1', 'flatten', 'dropout2', 'dense1']

if __name__ == "__main__":
    model = build_model()
    model.summary()