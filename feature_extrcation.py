from skimage.feature import hog
from preprocessing import process_image, process_video

def feature_extraction(process_image):
    process_image(dataset_path)
    
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    pass
    return features