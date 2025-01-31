import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def load_data(dataset="mnist"):
    if dataset == "mnist":
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        return train_x, train_y, test_x, test_y

def preprocess_data(data):
    train_x, train_y, test_x, test_y = data

    # Normalize
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    # Reshape
    train_x = np.expand_dims(train_x, -1)
    test_x = np.expand_dims(test_x, -1)

    # Split into train/val
    val_x = train_x[-10000:]
    val_y = train_y[-10000:]
    train_x = train_x[:-10000]
    train_y = train_y[:-10000]

    return train_x, val_x, test_x, train_y, val_y, test_y

def create_experiment_dir(base_dir, experiment_name):
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def show_confusion_matrix(matrix, class_names):
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center')
    plt.show()
