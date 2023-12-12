# PyTorch-Lightning-MNIST-Classifier

This repository contains a PyTorch Lightning implementation of a Multilayer Perceptron (MLP) for classifying MNIST digits. The code is structured to leverage PyTorch Lightning's features for a more organized and optimized deep learning workflow.

## Overview

The implementation includes defining a custom `Dataset`, a `DataModule` for handling data operations, and an `MLP_MNIST_Classifier` class for the MLP model. The model is trained and validated on the MNIST dataset, which is a large database of handwritten digits commonly used for training various image processing systems.

### Key Features

- **Custom Dataset**: The MNIST dataset is loaded and preprocessed into a custom PyTorch `Dataset`, enabling more control over data manipulation.
- **DataModule**: PyTorch Lightning's `DataModule` organizes data loading, splitting, and preprocessing, making it easy to manage the data pipeline.
- **Model**: The `MLP_MNIST_Classifier` is a simple yet effective neural network with three fully connected layers, implemented as a subclass of `pl.LightningModule`.
- **Training and Validation**: The training process includes both training and validation steps, with early stopping based on validation loss.

## Requirements

- Python 3.x
- PyTorch
- PyTorch Lightning
- NumPy

## Dataset

The MNIST dataset used in this implementation is expected to be in NumPy array format. The dataset is split into training, validation, and test sets. The `MNISTDataset` class handles the loading and preprocessing of this data.

## Model Architecture

The `MLP_MNIST_Classifier` consists of:
- An input layer of size 784 (flattened 28x28 images).
- Two hidden layers with 512 and 256 units, respectively, and ReLU activations.
- An output layer with 10 units (number of classes in MNIST), followed by a softmax layer for classification.

## Usage

To train and test the model, run the script as follows:

```bash
python mnist_classifier.py
```

### Customization

You can customize various parameters such as the batch size and learning rate through command line arguments.

## Results

The training process logs various metrics like training loss, validation accuracy, and validation loss. The final test results are printed at the end of the training.

## Contributing

Contributions to improve the code or extend its functionality are welcome. Please submit a pull request or open an issue for discussion.

## Referecnes

CNU ISoft Lab : 지능 소프트웨어 연구실
