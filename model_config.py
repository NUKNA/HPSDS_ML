from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Input, BatchNormalization

def build_model():
    model = Sequential([
        Input(shape=(800, 1)),
        Conv1D(8, kernel_size=200, strides=100, activation='relu', padding='same', name='conv1'),
        Flatten(name='flatten'),
        BatchNormalization(name='bn_conv1'),
        Dense(1, activation='linear', name='output'),
    ])
    return model

def layer_name():
    return ['conv1', 'flatten', 'output']

if __name__ == "__main__":
    model = build_model()
    model.summary()