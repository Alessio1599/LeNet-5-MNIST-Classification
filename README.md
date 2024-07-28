# MNIST classification using LeNet5 model
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This repository showcases a deep learning project implementing the LeNet-5 architecture [1]. The project utilizes the MNIST dataset to demonstrate LeNet-5's effectiveness in image classification.


## Table of Contents
<ul>
<li><a href="#running-the-project">Running the Project</a></li>
<li><a href="#directory-structure">Directory Structure</a></li>
<li><a href="#dataset">Dataset</a></li>
<li><a href="#lenet-5-model">LeNet-5 Model</a></li>
<li><a href="#utilities">Utilities</a></li>
<li><a href="#references">References</a></li>
</ul>

## Running the Project
1. Clone the repository:
    ```bash
    git clone https//github.com/Alessio1599/LeNet-5-MNIST-Classification.git
    cd LeNet-5-MNIST-Classification
    ```
2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the main script:
    ```bash
    python main.py
    ```
    Example of the use of flags
    ```bash
    python main.py --epochs=10 --batch_size=250
    ```

## Directory structure
```
LeNet-5-MNIST-Classification/
│
│
├── utils/
│   └── utils_DL.py
│   └── utils_data.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Dataset
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## References
[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998
