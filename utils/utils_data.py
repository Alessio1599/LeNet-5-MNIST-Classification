import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import datasets

def load_data():
    """
    Load the MNIST dataset.

    Returns:
    tuple: Tuple containing training and test data in the format ((train_x, train_y), (test_x, test_y)).
    """
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    class_names = range(10)  # the 10 digits
    return train_x, train_y, test_x, test_y, class_names

def preprocess_data(data):
    """
    Preprocess the MNIST dataset.

    Parameters:
    data (tuple): Tuple containing training and test data in the format ((train_x, train_y), (test_x, test_y)).

    Returns:
    tuple: Processed training, validation, and test datasets.
    """
    
    # Split the training set into validation and training sets
    val_size = 10000
    train_x, val_x, train_y, val_y = train_test_split(data[0], data[1], test_size=val_size, random_state=42, shuffle=True)
    test_x, test_y = data[2], data[3]

    # Add dimension -> (28,28,1)
    if len(train_x.shape) == 3:
        train_x = np.expand_dims(train_x, axis=3)
        val_x = np.expand_dims(val_x, axis=3)
        test_x = np.expand_dims(test_x, axis=3)

    # Normalize intensity range [0,255] -> [0,1]
    train_x, val_x, test_x = train_x / 255.0, val_x / 255.0, test_x / 255.0

    # Pad images to 32x32
    if train_x.shape[1] < 32 or train_x.shape[2] < 32:
        pad_h = int((32 - train_x.shape[1]) / 2)
        pad_w = int((32 - train_x.shape[2]) / 2)
        train_x = np.pad(train_x, ((0, 0), (pad_w, pad_w), (pad_h, pad_h), (0, 0)), 'constant', constant_values=0)
        val_x = np.pad(val_x, ((0, 0), (pad_w, pad_w), (pad_h, pad_h), (0, 0)), 'constant', constant_values=0)
        test_x = np.pad(test_x, ((0, 0), (pad_w, pad_w), (pad_h, pad_h), (0, 0)), 'constant', constant_values=0)

    return train_x, val_x, test_x, train_y, val_y, test_y

def inspect_data(Images, data_train_y, class_names): 
    """This function visualizes 10 random images from the given dataset.

    Args:
        Images (numpy.ndarray): A numpy array containing image data.
        data_train_y (numpy.ndarray): A numpy array containing image labels.
        class_names (list): A list of class names corresponding to the image labels.

    This function displays a visual inspection of 10 randomly selected images 
    from the provided dataset. It assumes that the images are grayscale and 
    arranges them in a horizontal layout with no axis labels. Additionally, it 
    assigns titles to each displayed image based on the corresponding class name 
    retrieved from the `class_names` list or array.
    """
    # Visualization of 10 random images of the dataset
    image_count=10 # quante immagini visualizzare
    _, axs = plt.subplots(1, image_count,figsize=(15, 10))
    for i in range(image_count):
        random_idx=random.randint(0,Images.shape[0]-1)
        axs[i].imshow(Images[random_idx],cmap='gray') #I can remove cmap='gray' if I want to see the color images
        axs[i].axis('off')
        axs[i].set_title(class_names[data_train_y[random_idx]])
    plt.show()
    