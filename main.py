

from preprocessing import preprocess_images, process_video
from feature_extrcation import extract_features, visualize_features

def main():
    dataset_path = r'D:\SHU\ML_lab\Assesment\Number_Test_Data'

    preprocessed_images = preprocess_images(dataset_path)
    hog_features, hog_images= extract_features(preprocessed_images)
    visualize_features(preprocessed_images, hog_images)

    # process_video(dataset_path)

if __name__ == "__main__":
    main()
