# Flowers Classification
Flowers Classifier is a project that develops a convolutional neural network using MobileNet to classify 102 types of flowers, providing an interactive web interface for users to upload and classify flower images.
## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Contact](#contact)

## Introduction
This project aims to classify 102 different types of flowers using a deep learning model. The model is trained using a convolutional neural network with MobileNet as the base architecture. The web interface allows users to upload images and get predictions on the type of flower.

![flowers_classification](https://github.com/UlianaEzubchik/Flowers_classification/assets/88460922/51ad5dbc-df5c-410a-921c-4141be113e89)

## Features
- **Deep Learning Model**: Uses MobileNet for efficient image classification.
- **Data Augmentation**: Enhances the training dataset with various transformations.
- **Interactive Web Interface**: Built with Streamlit for easy image upload and classification.
- **Early Stopping and Learning Rate Reduction**: Improves training efficiency and model performance.

## Installation

### Prerequisites
- [Python 3.6+](https://www.python.org/downloads/)
- [TensorFlow](https://www.tensorflow.org/install)
- [Keras](https://keras.io/getting_started/)
- [Streamlit](https://docs.streamlit.io/library/get-started/installation)
- [NumPy](https://numpy.org/install/)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
- [Matplotlib](https://matplotlib.org/stable/users/installing.html)


### Steps
1. Clone the repository:
    ```
    git clone https://github.com/UlianaEzubchik/Flowers_classification.gitt
    cd flowers-classifier
    ```

2. Install the required packages:
    ```
    pip install tensorflow keras streamlit numpy pandas matplotlib
    ```

3. Download and prepare the dataset:
    - Ensure you have the dataset in the `dataset/train`, `dataset/test`, and `dataset/valid` directories.
    - Ensure `cat_to_name.json` is in the root directory of the project.
  
## Usage

### Web Interface
To start the web interface for flower classification, run:
```
streamlit run app.py
```
- Upload an image of a flower to the web interface.
- The application will classify the image and display the predicted flower category and confidence score

## Model Training
The model training script (Flowers classifier with data augmentation.ipynb) includes:
  - Data augmentation for the training dataset.
  - Definition of the MobileNet-based model architecture.
  - Training the model with early stopping and learning rate reduction callbacks.
  - Saving the best model based on validation performance. 

## Deployment
The web interface is built using Streamlit, allowing users to upload images for classification. The deployment script (app.py) includes:
- Loading the trained model.
- Preprocessing the uploaded image.
- Predicting the flower category and displaying the results.

## Contact
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/ulyana-yezubchyk/)](https://www.linkedin.com/in/ulyana-yezubchyk/)
[![Email](https://img.shields.io/badge/Email-ulyaa.071@gmail.com-green.svg)](mailto:your_email@example.com)

[![Back to Top](https://img.shields.io/badge/-Back_to_Top-blue?style=flat-square)](#Flowers-Classification)
