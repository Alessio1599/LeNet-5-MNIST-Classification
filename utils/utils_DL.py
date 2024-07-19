#utils/utils_DL.py
import matplotlib.pyplot as plt
import numpy as np
from models.model import build_lenet5
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
  history = model.fit(train_x, train_y, epochs=5, batch_size=100, validation_data=(val_x, val_y), verbose=1)
  
  # Plot of the training history
  plot_history(history)

  return model

def print_training_summary(history):
  """
  Prints the training summary including the hyperparameters and the loss function.
  """
  print("\nTraining Summary:")
  print(f"Number of epochs: {len(history.epoch)}")
  print(f"Batch size: {history.params['batch_size']}")
  print(f"Optimizer: {history.model.optimizer._name}")
  print(f"Loss function: {history.model.loss}")

def evaluate_model(model, test_x, test_y, class_names):
  # Evaluation of the model
  test_loss, test_accuracy = model.evaluate(test_x, test_y)
  print('Test loss: ', test_loss)
  print('Test accuracy: ', test_accuracy)
  
  # Confusion matrix
  y_pred = model.predict(test_x)
  y_pred = np.argmax(y_pred, axis=1)
  show_confusion_matrix(confusion_matrix(test_y, y_pred), class_names)

def plot_history(history,metric=None):
  """
  Draws in a graph the loss trend over epochs on both training and validation sets. Moreover, if provided, it draws in the same graph also the trend of the given metric.
  """
  fig, ax1 = plt.subplots(figsize=(10, 8))

  epoch_count=len(history.history['loss'])

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
  img=ax.matshow(conf_matrix)
  tick_marks = np.arange(len(class_names))
  _=plt.xticks(tick_marks, class_names,rotation=45)
  _=plt.yticks(tick_marks, class_names)
  _=plt.ylabel('Real')
  _=plt.xlabel('Predicted')

  for i in range(len(class_names)):
    for j in range(len(class_names)):
        text = ax.text(j, i, '{0:.1%}'.format(conf_matrix[i, j]),
                       ha='center', va='center', color='w')
