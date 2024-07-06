import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import sys
sys.path.append('/Users/alessioguarachi/Desktop/Python codes')
from DL_models.utils_DL import plot_history,show_confusion_matrix
from Images.utils_Images import Image_inspection

# Upload of a dataset, in example ther MNIST
(data_train_x,data_train_y), (data_test_x,data_test_y) = keras.datasets.mnist.load_data()
class_names=range(10) # the 10 digits

class_count=len(class_names)
print('Type of data_train_x',type(data_train_x))# Numpy array
print('Training dataset shape',data_train_x.shape) #60000,28,28

Image_inspection(data_train_x) #Show 10 random images

# Split of the training set in validation and training set
val_size= 10000 # We can pass the absolute value or the percentage
train_x, val_x, train_y, val_y = train_test_split(data_train_x, data_train_y, test_size=val_size, random_state=42,shuffle=True) # If ypu don't specity the amount of splitting, it will use the default one
print('Final shape of the training dataset',train_x.shape) #We have 50000 images 28x28

test_x=data_test_x
test_y=data_test_y

#Since each image is (28,28) it is necessary to add a dimension -> (28,28,1)
if (len(train_x.shape)==3): # se la dimensione è 3 aggiungiamo una dimensione
  train_x=np.expand_dims(train_x,axis=3)
  val_x=np.expand_dims(val_x,axis=3)
  test_x=np.expand_dims(test_x,axis=3)
  print('Train shape: ',train_x.shape)
  print('Validation shape: ',val_x.shape)
  print('Test shape: ',test_x.shape)

# Intensity range 
print('Min value: ',train_x.min()) #0
print('Max value: ',train_x.max()) #255 Sono stati utilizzati 8 bit (1 byte) per codificare ogni pixel dell'immagine

#Intensity range normalization [0,255] -> [0,1]
train_x=train_x/255 # (Example 255/255=1, for the maximum value instead 0/255=0)
val_x=val_x/255
test_x=test_x/255
print('Min value: ',train_x.min())
print('Max value: ',train_x.max())

# To mantain the original input shape of the original LeNet-5 we have to pad the images
if train_x.shape[1]<32 or train_x.shape[2]<32: #in train_x.shape[0] we have the number of images
  pad_h=int((32-train_x.shape[1])/2)
  pad_w=int((32-train_x.shape[2])/2)
  train_x=np.pad(train_x,((0,0),(pad_w,pad_w),(pad_h,pad_h),(0,0)),'constant',constant_values=0)
  val_x=np.pad(val_x,((0,0),(pad_w,pad_w),(pad_h,pad_h),(0,0)),'constant',constant_values=0)
  test_x=np.pad(test_x,((0,0),(pad_w,pad_w),(pad_h,pad_h),(0,0)),'constant',constant_values=0)
  print('Train shape: ',train_x.shape)
  print('Validation shape: ',val_x.shape)
  print('Test shape: ',test_x.shape)

# Build and compile the model
model=build_lenet5()
model.summary()

# Train the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_x,train_y,epochs=10,batch_size=64,validation_data=(val_x,val_y))

#Plot of the training history
plot_history(history)

#Evaluation of the model
test_loss, test_accuracy = model.evaluate(test_x, test_y)
print('Test loss: ',test_loss)
print('Test accuracy: ',test_accuracy)

#Confusion matrix
y_pred=model.predict(test_x)
y_pred=np.argmax(y_pred,axis=1)
show_confusion_matrix(test_y,y_pred,class_names)

#Prediction
index=0
image=test_x[index]
label=test_y[index]
image=np.expand_dims(image,axis=0)
prediction=model.predict(image)
prediction=np.argmax(prediction)
print('Label: ',label)
print('Prediction: ',prediction)

#Show the image
plt.imshow(image[0,:,:,0],cmap='gray')
plt.show()


# Show the filters of the first layer
layer=model.get_layer('C1')
