# models/model.py
import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_lenet5(input_shape=(32, 32, 1), output_class_count=10):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape, name='Input'),
            layers.Conv2D(filters=6, kernel_size=5, strides=1, activation='tanh', padding='valid', name='C1'),
            layers.AvgPool2D(pool_size=2, strides=2, name='S2'),
            layers.Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh', padding='valid', name='C3'),
            layers.AvgPool2D(pool_size=2, strides=2, name='S4'),
            layers.Conv2D(filters=120, kernel_size=5, strides=1, activation='tanh', padding='valid', name='C5'),
            layers.Flatten(),
            layers.Dense(84, activation='tanh', name='F6'),
            layers.Dense(units=output_class_count, activation='softmax', name='Output')
        ]
    )
    return model
