import numpy as np  
from tensorflow import keras
from sklearn.model_selection import train_test_split

class MNISTDataHandler:
    def __init__(self, val_size=0.2):
        self.val_size = val_size
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_mnist()
        self.x_train, self.x_val, self.y_train, self.y_val = self.split_data(self.x_train, self.y_train)
        self.x_train = self.normalize_data(self.x_train)
        self.x_val = self.normalize_data(self.x_val)
        self.x_test = self.normalize_data(self.x_test)
        self.x_train = self.expand_dims(self.x_train)
        self.x_val = self.expand_dims(self.x_val)
        self.x_test = self.expand_dims(self.x_test)

    def load_mnist(self):
        (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
        return train_x, train_y, test_x, test_y

    def split_data(self, x, y):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.val_size)
        return x_train, x_val, y_train, y_val

    def normalize_data(self, x):
        # Intensity range normalization
        x = x / 255.0
        return x
    
    def expand_dims(self, x):
        if (len(x.shape) == 3):
            x = np.expand_dims(x, axis=3)
        return x
