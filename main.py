

import os
from preprocessing import preprocess_images, process_video
from pre_processFunctions import custom_processes, process_and_extract_digits, zero_nine_images
# from feature_extrcation
import numpy as np

def main():
    dataset_path = r'D:\SHU\ML_lab\Assesment\Number_Test_Data'
    all_images = [img for img in os.listdir(dataset_path) if img.endswith(".png")]
    image_path = [os.path.join(dataset_path, img_path) for img_path in all_images]
    print(image_path)
    all_digits = []
    raw_images = []
    image_name = [os.path.basename(path) for path in image_path]
    print(image_name)

     # Process image and extract digits
    for img, image in zip(image_name, image_path):
        if img.startswith(tuple(f"{i:03}" for i in range(10))):  # First 10 images (000.png to 009.png)
            print(f"Processing raw image {img} without preprocessing")
            raw_image = zero_nine_images(image)  # Return raw image
            raw_images.append(raw_image)  # Store raw images for training
            continue

        elif img == '013.png':
            print("Processing image 13 with custom process")
            process = custom_processes["image_13"]

        elif img == '014.png':
            print("Processing image 14 with custom process")
            process = custom_processes["image_14"]

        else:
            process = custom_processes["image_0_9"]

        # Process the image with the correct pipeline
        digits = process_and_extract_digits(image, process)
       
        # print(f"Extracted digits: {digits}")

        if digits is not None and len(digits) > 0:  # Ensure digits are valid
            all_digits.append(digits)  # Store extracted digits

    print("Processing complete.")
    return raw_images, all_digits


    
    # print(preprocessed_images)


if __name__ == "__main__":
    raw_images, all_digits = main()
    print(f"Number of raw images: {len(raw_images)}")
    print(f"Number of digit sets: {len(all_digits)}")
