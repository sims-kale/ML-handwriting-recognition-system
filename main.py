

from preprocessing import preprocess_images, process_video
from feature_extrcation import extract_features, visualize_features, extract_video_features
import numpy as np

def main():
    dataset_path = r'D:\SHU\ML_lab\Assesment\Number_Test_Data'

    preprocessed_images = preprocess_images(dataset_path)
    hog_features, hog_images, edges_images = extract_features(preprocessed_images)
    # Step 4: Convert Features to NumPy Array
    hog_features = np.array(hog_features)  # Ensure correct format
    labels = np.arange(len(hog_features))  # Generate dummy labels (Replace with actual labels if available)
    visualize_features(preprocessed_images, hog_images, edges_images)


    #Video 
    process_video(dataset_path)
    extract_video_features()
    
    


    # process_video(dataset_path)

if __name__ == "__main__":
    main()
