import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from models.model import build_lenet5
from utils.utils_DL import plot_history, show_confusion_matrix
from utils.utils_data import load_data, inspect_data, preprocess_data

def main():
    data_train_x, data_train_y, data_test_x, data_test_y, class_names=load_data()
    #print('Type of data_train_x', type(data_train_x))  # Numpy array
    #print('Training dataset shape', data_train_x.shape)  # 60000, 28, 28
    
    # Inspect the data
    inspect_data(data_train_x, class_names)

    train_x, val_x, test_x, train_y, val_y, test_y = preprocess_data((data_train_x, data_train_y, data_test_x, data_test_y))
    #print('Final shape of the training dataset', train_x.shape)  # We have 50000 images 28x28
    #print('Train shape: ', train_x.shape)
    #print('Validation shape: ', val_x.shape)
    #print('Test shape: ', test_x.shape)

    # Build and compile the model
    model = build_lenet5()
    model.summary()

    # Train the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_x, train_y, epochs=10, batch_size=64, validation_data=(val_x, val_y))

    # Plot of the training history
    plot_history(history)

    # Evaluation of the model
    test_loss, test_accuracy = model.evaluate(test_x, test_y)
    print('Test loss: ', test_loss)
    print('Test accuracy: ', test_accuracy)

    # Confusion matrix
    y_pred = model.predict(test_x)
    y_pred = np.argmax(y_pred, axis=1)
    show_confusion_matrix(confusion_matrix(test_y, y_pred), class_names)

if __name__ == '__main__':
    main()
