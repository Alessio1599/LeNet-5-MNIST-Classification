import matplotlib.pyplot as plt
import numpy as np
import random

def Image_inspection(Images): #def Image_inspection(Images,labels,class_names):
    """This function visualizes 10 random images from the given dataset.

    Args:
        dataset (numpy.ndarray): A numpy array containing image data.

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
        random_idx=random.randint(0,Images.shape[0])
        axs[i].imshow(Images[random_idx],cmap='gray') #I can remove cmap='gray' if I want to see the color images
        axs[i].axis('off')
        #axs[i].set_title(class_names[class_names[random_idx]])
    