

import os
from preprocessing import preprocess_images, process_video
from pre_processFunctions import custom_processes, process_and_extract_digits
# from feature_extrcation
import numpy as np

def main():
    dataset_path = r'D:\SHU\ML_lab\Assesment\Number_Test_Data'
    all_images = [img for img in os.listdir(dataset_path) if img.endswith(".png")]
    image_path = [os.path.join(dataset_path, img_path) for img_path in all_images[13:14]]
    print(image_path)
    all_digits = []
    image_name  = [os.path.basename(path) for path in image_path]
    print(image_name)
    

    if '013.png' in image_name:
        print("Processing image 14 with custom process")
        process = custom_processes["image_14"]

    else:
        process = custom_processes["default"]

    # Process image and extract digits
    digits = process_and_extract_digits(image_path, process)
    # preprocessed_images = preprocess_images(img_path)

    if digits:
        all_digits.append(digits)  # Store extracted digits

    print("Processing complete.")
    return all_digits


    
    # print(preprocessed_images)


if __name__ == "__main__":
    main()
