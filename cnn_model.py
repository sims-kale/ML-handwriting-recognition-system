import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

# Load MNIST dataset
def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32')/ 255.0
    x_test = x_test.astype('float32')/ 255.0
    x_train = x_train.reshape(-1, 28,28,1)
    x_test = x_test.reshape(-1,28,28,1)


    # One-hot encode the labels
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test

# Define CNN model
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')  # Output layer for 10 classes
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
              loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Train and save model
def train_and_save_model():
    x_train, y_train, x_test, y_test = load_mnist()
    model = create_model()
    model.fit(x_train, y_train, epochs=10, batch_size = 64, validation_data=(x_test, y_test))
    model.save("mnist_model.h5")  # Save model for later use
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    return model

if __name__ == "__main__":
    train_and_save_model()
