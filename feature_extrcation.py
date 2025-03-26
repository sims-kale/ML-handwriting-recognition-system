import os
import matplotlib.pyplot as plt
from skimage.feature import hog
import numpy as np
import cv2

def extract_features(process_images):
    pass
def extract_video_features():
    hog_features = []
    edges_images = []
    hog_images = []
    video_frames_path = r'D:\SHU\ML_lab\Assesment\Number_Test_Data\video_frames'

    for frame_name in os.listdir(video_frames_path):
        frame_path = os.path.join(video_frames_path, frame_name)
        
        # Read frame as grayscale
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Skipping {frame_name} (not a valid image)")
            continue

        # Extract HOG features
        features, hog_image = hog(frame, pixels_per_cell=(4, 4), 
                                  cells_per_block=(3, 3), visualize=True)
        hog_features.append(features)
        hog_images.append(hog_image)

        # Apply Edge Detection (Canny)
        edges = cv2.Canny(frame, 100, 200)
        edges_images.append(edges)

    return hog_features, hog_images, edges_images


def visualize_features(process_images, hog_images, edges_images):
    f, axes = plt.subplots(3, len(process_images), figsize=(15, 7))

    for i in range(len(process_images)):
        axes[0, i].imshow(process_images[i], cmap='gray')  # Preprocessed image
        axes[0, i].axis('off')

        axes[1, i].imshow(hog_images[i], cmap='gray')  # HOG Features
        axes[1, i].axis('off')

        axes[2, i].imshow(edges_images[i], cmap='gray')  # Edge Detection
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()