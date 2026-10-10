from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D

def build_model():
    model = Sequential([
        Conv1D(32, kernel_size=16, strides=2, activation='relu', padding='same', input_shape=(800, 1)),
        Conv1D(64, 8, strides=1, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')
    ])
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()