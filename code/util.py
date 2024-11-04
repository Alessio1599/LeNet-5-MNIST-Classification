#code/util.py

##utils/utils_data.py
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
    
#utils/utils_DL.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

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

def train_model(train_x, train_y, val_x, val_y):
  # Build and compile the model
  model = build_lenet5()
  model.summary()

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  # Train the model
  history = model.fit(train_x, train_y, epochs=5, batch_size=1000, validation_data=(val_x, val_y), verbose=1)
  
  # Plot of the training history
  plot_history(history)

  return model

def print_training_summary(history):
  """
  Prints the training summary including the hyperparameters and the loss function.
  """
  print("\nTraining Summary:")
  print(f"Number of epochs: {history.params['epochs']}")
  print(f"Optimizer: {history.model.optimizer.name}")
  print(f"Loss function: {history.model.loss}")

def evaluate_model(model, test_x, test_y, class_names):
  # Evaluation of the model
  test_loss, test_accuracy = model.evaluate(test_x, test_y)
  print('Test loss: ', test_loss)
  print('Test accuracy: ', test_accuracy)
  
  # Confusion matrix
  test_conf_pred = model.predict(test_x) # Predicted probabilities
  test_y_pred = np.argsort(test_conf_pred,axis=1)[:,-1] # Predicted classes, the one with the highest probability
  conf_matrix = confusion_matrix(test_y, test_y_pred, normalize='true')
  show_confusion_matrix(conf_matrix, class_names)
  print_training_summary(model.history)

def plot_history(history,metric=None):
  """
  Draws in a graph the loss trend over epochs on both training and validation sets. Moreover, if provided, it draws in the same graph also the trend of the given metric.
  """
  fig, ax1 = plt.subplots(figsize=(10, 8))

  epoch_count = len(history.history['loss'])

  line1,=ax1.plot(range(1,epoch_count+1),history.history['loss'],label='train_loss',color='orange')
  ax1.plot(range(1,epoch_count+1),history.history['val_loss'],label='val_loss',color = line1.get_color(), linestyle = '--')
  ax1.set_xlim([1,epoch_count])
  ax1.set_ylim([0, max(max(history.history['loss']),max(history.history['val_loss']))])
  ax1.set_ylabel('loss',color = line1.get_color())
  ax1.tick_params(axis='y', labelcolor=line1.get_color())
  ax1.set_xlabel('Epochs')
  _=ax1.legend(loc='lower left')

  if (metric!=None):
    ax2 = ax1.twinx()
    line2,=ax2.plot(range(1,epoch_count+1),history.history[metric],label='train_'+metric)
    ax2.plot(range(1,epoch_count+1),history.history['val_'+metric],label='val_'+metric,color = line2.get_color(), linestyle = '--')
    ax2.set_ylim([0, max(max(history.history[metric]),max(history.history['val_'+metric]))])
    ax2.set_ylabel(metric,color=line2.get_color())
    ax2.tick_params(axis='y', labelcolor=line2.get_color())
    _=ax2.legend(loc='upper right')

def show_confusion_matrix(conf_matrix,class_names,figsize=(10,10)):
  fig, ax = plt.subplots(figsize=figsize)
  img = ax.matshow(conf_matrix)
  tick_marks = np.arange(len(class_names))
  _=plt.xticks(tick_marks, class_names,rotation=45)
  _=plt.yticks(tick_marks, class_names)
  _=plt.ylabel('Real')
  _=plt.xlabel('Predicted')

  for i in range(len(class_names)):
    for j in range(len(class_names)):
        text = ax.text(j, i, '{0:.1%}'.format(conf_matrix[i, j]),
                       ha='center', va='center', color='w')
  plt.show()
