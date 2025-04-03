# Handwriting Recognition System (0-9)

This repository contains a notebook that implements a handwriting recognition system using Machine Learning and Computer Vision techniques. The core model is built using TensorFlow/Keras and designed to classify handwritten digits (0-9) using a Convolutional Neural Network (CNN) and Residual Neural Network (ResNet)

## Overview
- **Purpose:** Build and evaluate a model for handwritten digit recognition.
- **Key Techniques:** CNN & ResNet architecture, data preprocessing, model evaluation (confusion matrix, classification report, etc.).
- **Notebook Structure:** The notebook includes sections for model building, training, evaluation, visualization, data loading, preprocessing, and Predictions.

## How to Run
1. **Install Dependencies:**  
   Make sure to run `pip install -r requirements.txt` to install all necessary libraries.
2. **Run the Notebook:**  
   Execute the cells sequentially to build, train, and evaluate the model.

## Additional Resources
This repository also includes a `utils` folder containing auxiliary files used during development and testing. These files are for preliminary work and rough experiments:
- `preprocessing.py` – Contains data preprocessing functions and a custom process for different types of images.
- `main.py` – A script used for testing preprocessing components with custom processes.
- `cnn_model.py` – Code for building and testing the CNN model.
- `resNet_model.py` – Code for experimenting with a ResNet architecture.

These resources have been invaluable for testing different approaches and refining the final notebook implementation.

## Results
The notebook demonstrates the model's performance through training metrics and visualizations, showcasing the effectiveness of the CNN & ResNet for digit recognition.

Enjoy exploring the project!🥂
