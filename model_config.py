from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as lyrs

def build_model(input_shape:int):
    model = Sequential([
        lyrs.Input(shape=(input_shape,1)),
        lyrs.Conv1D(32, kernel_size=16, strides=2, activation='relu', padding='same', name='conv1'),
        lyrs.Conv1D(64, 8, strides=1, activation='relu', padding='same', name='conv2'),
        lyrs.MaxPooling1D(pool_size=2, name='maxpool'),
        lyrs.Dropout(0.25 , name='dropout1'),
        lyrs.Flatten(name='flatten'),
        lyrs.Dense(128, activation='relu', name='dense1'),
        lyrs.Dropout(0.5, name='dropout2'),
        lyrs.Dense(1, activation='linear')
    ])
    return model

def layer_name():
    return ['conv1', 'conv2', 'maxpool', 'dropout1', 'flatten', 'dropout2', 'dense1']

if __name__ == "__main__":
    model = build_model(670-460)
    model.summary()