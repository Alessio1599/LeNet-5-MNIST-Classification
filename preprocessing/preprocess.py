import numpy as np
from sklearn.model_selection import train_test_split

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
