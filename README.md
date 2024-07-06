# Deep-Learning
The repository contains a project that uses deep learning to 
LeNet-5 [LeCun et al., 1998]


## Table of Contents
<ul>
<li><a href="#References">References</a></li>
<li><a href="#Dataset">Dataset</a></li>
</ul>

## Running the Project
1. Clone the repository:
    ```bash
    git clone https://github.com/Alessio1599/Deep-Learning.git
    cd Deep-Learning
    ```
2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the main script:
    ```bash
    python main.py
    ```

## Directory structure
Deep-Learning/
│
├── models/
│   └── model.py
│
├── preprocessing/
│   └── preprocess.py
│
├── utils/
│   └── utils_DL.py
│   └── utils_Images.py
│
├── main.py
├── requirements.txt
└── README.md


# Dataset
The project uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which is automatically downloaded when running the scripts or notebooks.

## LeNet-5 Model
The model is built using the LeNet-5 architecture, which consists of:
- Convolutional layers
- Average pooling layers
- Fully connected layers

## Utilities
The utility functions help in visualizing the training process and evaluating the model performance:
- `plot_history(history, metric)`: Plots the training and validation loss and metrics.
- `show_confusion_matrix(conf_matrix, class_names)`: Displays the confusion matrix with class labels.
- `Image_inspection(data)`: Displays 10 random images from the dataset.

# References
[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998
