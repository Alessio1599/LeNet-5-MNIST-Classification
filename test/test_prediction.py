"""
After training the model, we can use it to make predictions on new data.
"""
import numpy as np
import matplotlib.pyplot as plt
    
def predict_image(model, test_x, test_y, index):
    image = test_x[index]
    label = test_y[index]
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    print('Label: ', label)
    print('Prediction: ', prediction)

    # Show the image
    plt.imshow(image[0, :, :, 0], cmap='gray')
    plt.title(f'Label: {label}, Prediction: {prediction}')
    plt.show()
    
# For demonstration purposes, predict the first image in the test set
# predict_image(model, test_x, test_y, 0)