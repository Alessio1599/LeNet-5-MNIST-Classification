from tensorflow import keras

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f"x_train shape: {x_train.shape}")

class_names = range(10)  # the 10 digits