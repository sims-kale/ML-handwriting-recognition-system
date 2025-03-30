import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
# ResNet-like architecture
def create_resnet_like_model():
    inputs = keras.Input(shape=(28, 28, 1))  # Ensure input shape matches (28, 28, 1) for MNIST

    # Convolutional Block
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # Residual Block 1
    shortcut = x
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, shortcut])  # Add shortcut connection
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # Residual Block 2
    shortcut = keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding="same")(x)  # Adjust shortcut
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, shortcut])  # Add shortcut connection
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # Fully Connected Layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)  # Output layer for 10 classes

    # Create the model
    resnet_model = keras.Model(inputs, outputs, name="ResNet_Like_Numbers_Recognition_Model")
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
    )
    resnet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Plot the model architecture and save it as an image
    keras.utils.plot_model(
        resnet_model,
        show_shapes=True,
        show_layer_names=True, 
    )

    # Display the model summary
    resnet_model.summary()
    return resnet_model

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, axis=-1)  
x_test = np.expand_dims(x_test, axis=-1)    
y_train = keras.utils.to_categorical(y_train, 10)  
y_test = keras.utils.to_categorical(y_test, 10)    

# Train and save the second model
resnet_model = create_resnet_like_model()
resnet_history = resnet_model.fit(
    x_train, y_train, epochs=8, batch_size=64, validation_data=(x_test, y_test)
)
resnet_model.save("resnet_like_model.keras")
resnet_test_loss, resnet_test_acc = resnet_model.evaluate(x_test, y_test)
print(f"ResNet-Like Model Test Accuracy: {resnet_test_acc * 100:.2f}%")