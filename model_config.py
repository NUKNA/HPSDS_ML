from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as lyrs

def build_model(input_shape:int):
    model = Sequential([
        lyrs.Input(shape=(input_shape, 1)),
        lyrs.Conv1D(8, kernel_size=200, strides=2, padding='same', name='conv1'),
        lyrs.BatchNormalization(name='bn_conv1'),
        lyrs.Activation('relu', name='activation_conv1'),
        lyrs.MaxPooling1D(name='max_pooling1'),

        lyrs.Conv1D(16, kernel_size=100, strides=1, padding='same', name='conv2'),
        lyrs.BatchNormalization(name='bn_conv2'),
        lyrs.Activation('relu', name='activation_conv2'),
        lyrs.MaxPooling1D(name='max_pooling2'),

        lyrs.Conv1D(32, kernel_size=20, strides=1, padding='same', name='conv3'),
        lyrs.BatchNormalization(name='bn_conv3'),
        lyrs.Activation('relu', name='activation_conv3'),
        lyrs.MaxPooling1D(name='max_pooling3'),

        lyrs.Flatten(name='flatten1'),
        lyrs.Dense(800, name='dense1'),
        lyrs.Dropout(0.2, name='dropout1'),
        lyrs.Dense(1, activation='linear', name='output'),
    ])
    return model

def layer_name():
    return ['conv1','bn_conv1', 'activation_conv1', 'max_pooling1',
            'conv2','bn_conv2', 'activation_conv2', 'max_pooling2',
            'conv3','bn_conv3', 'activation_conv3', 'max_pooling3',
            'flatten1', 'dense1', 'dropout1', 'output']

if __name__ == "__main__":
    model = build_model(670-460)
    model.summary()