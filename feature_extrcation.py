import matplotlib.pyplot as plt
from skimage.feature import hog

def extract_features(process_images):
    hog_features = []
    hog_images = []

    for image in process_images:
        features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(features)
        hog_images.append(hog_image)

    return hog_features, hog_images

def visualize_features(process_images, hog_images):
    f, axes = plt.subplots(2, len(process_images), figsize=(18, 5))

    for i in range(len(process_images)):
        axes[0, i].imshow(process_images[i], cmap='gray')  # Preprocessed image
        axes[0, i].axis('off')

        axes[1, i].imshow(hog_images[i], cmap='gray')  # HOG Features
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
