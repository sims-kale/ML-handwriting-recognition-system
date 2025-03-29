import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from pre_processFunctions import custom_processes, process_and_extract_digits, zero_nine_images

def load_model_and_predict(preprocessed_images):
    """Loads trained CNN model and predicts digits from preprocessed images."""
    model = keras.models.load_model("mnist_model.h5")  # Load trained model
    predictions = []
    
    for img in preprocessed_images:
        # Ensure image is 28x28 and normalized
        
        # img = img / 255.0  # Normalize
        # img = cv2.resize(img, (28, 28))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(f"Loaded img shape: {img.shape}")

        img = img.reshape(1, 28, 28, 1) / 255.0
        print("Image shape before prediction:", img.shape)
        pred_probs = model.predict(img)  # Get probability distribution
        pred_digit = np.argmax(pred_probs)  # Get highest probability digit
        # print(f"Raw Model Output: {pred_probs}")  # Print confidence scores
        # print(f"Predicted Digit: {pred_digit}")

        predictions.append(pred_digit)
        plt.imshow(img.squeeze() , cmap="gray")
        plt.title(f"Prediction: {pred_digit}")
        plt.axis("off")
        plt.show()

    return predictions

def main():
    dataset_path = r'D:\SHU\ML_lab\Assesment\Number_Test_Data'
    all_images = [img for img in os.listdir(dataset_path) if img.endswith(".png")]
    image_path = [os.path.join(dataset_path, img_path) for img_path in all_images[10:11]]
    print(image_path)
    image_name = [os.path.basename(path) for path in image_path]
    print(image_name)

    # Dictionary to store predictions for each image
    image_predictions = {}

    # Process image and extract digits
    for img, image in zip(image_name, image_path):
        if img.startswith(tuple(f"{i:03}" for i in range(10))):  # First 10 images (000.png to 009.png)
            print(f"Processing raw image {img} without preprocessing")
            raw_image = zero_nine_images(image)  # Return raw image
            image_predictions[img] = load_model_and_predict([raw_image])  # Predict directly for raw images
            continue
        elif img == '010.png':
            print("Processing image 10 with Custome process")
            process = custom_processes["image_10"]
            

        elif img == '013.png':
            print("Processing image 13 with custom process")
            process = custom_processes["image_13"]

        elif img == '014.png':
            print("Processing image 14 with custom process")
            process = custom_processes["image_14"]

        else:
            process = custom_processes["image_0_9"]

        # Process the image using the selected preprocessing function
        digits = process_and_extract_digits(image, process)

        if digits is not None and len(digits) > 0:
            print(f"Number of digits extracted from {img}: {len(digits)}")
            
            # Predict digits for this image
            predicted_numbers = load_model_and_predict(digits)
            print("final predictions: ", predicted_numbers)
            image_predictions[img] = predicted_numbers  # Store predictions for this image
           
        else:
            print(f"No digits extracted from {img}.")

    for img_name, img_path in zip(image_name, image_path):
        original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            print(f"Error: Unable to read image {img_path}")
            continue

        # Get predictions for this specific image
        combined_prediction = ''.join(map(str, image_predictions.get(img_name, [])))

        # Display the original image with its predictions
        plt.figure(figsize=(6, 6))
        plt.imshow(original_image, cmap="gray")
        plt.title(f"{img_name}: {combined_prediction}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    # Optional: Train CNN if not already trained
    if not os.path.exists("mnist_model.h5"):
        print("Training CNN model...")
        os.system("python CNN_model.py")  # Runs CNN training only if model is missing

    main()