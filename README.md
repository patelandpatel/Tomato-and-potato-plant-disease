# 🌱 Tomato and Potato Plant Disease Detection 🌱

A deep learning project to detect and classify diseases in tomato and potato plants using image recognition techniques. This repository contains code, datasets, and model configurations for training and deploying a disease detection model aimed at assisting farmers and agricultural researchers.




## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project uses computer vision and deep learning to detect diseases in tomato and potato plants. The primary goal is to build a model that can classify common diseases from images of plant leaves, enabling early and efficient intervention in agricultural practices.

## Features

- Detects multiple diseases affecting tomato and potato plants.
- Uses a convolutional neural network (CNN) model for high accuracy.
- Supports real-time disease prediction through an interactive interface.
- Pre-trained weights and model configurations provided for quick setup.

## Dataset

The dataset used in this project is based on images of healthy and diseased plant leaves. The data includes images labeled with specific diseases like *Early Blight*, *Late Blight*, and *Healthy* for both tomato and potato plants. [Kaggle Plant Village Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) is recommended as a starting point.

## Requirements

- Python 3.6 or later
- TensorFlow or PyTorch
- OpenCV
- Pandas
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/patelandpatel/Tomato-and-Potato-Plant-Disease.git
   cd Tomato-and-Potato-Plant-Disease
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) or another source and place it in the `data/` directory.

## Usage

1. **Train the Model**  
   To train the model on your local machine, run the following command:
   ```bash
   python train.py --dataset data/plant_disease --epochs 50 --batch_size 32
   ```

2. **Evaluate the Model**  
   After training, evaluate the model on a test dataset:
   ```bash
   python evaluate.py --model saved_model.h5 --test_data data/test
   ```

3. **Run Inference**  
   Use the model to classify new images:
   ```bash
   python predict.py --image path_to_image.jpg --model saved_model.h5
   ```

## Model Architecture

The model is built using Convolutional Neural Networks (CNNs) with layers optimized for feature extraction from leaf images. A transfer learning approach using pre-trained models (e.g., ResNet, VGG) can also be applied for better results.

### Training Parameters

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Batch Size: 32
- Learning Rate: 0.001

## Results

The model achieved an accuracy of approximately **97%** on the test dataset. Below are some sample results:

| Disease            | Precision | Recall | F1-Score |
|--------------------|-----------|--------|----------|
| Early Blight       | 0.96      | 0.97   | 0.965    |
| Late Blight        | 0.95      | 0.96   | 0.955    |
| Healthy            | 0.98      | 0.99   | 0.985    |

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add a new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Create a pull request.

Please make sure to follow the [code of conduct](CODE_OF_CONDUCT.md) and adhere to the coding guidelines.
