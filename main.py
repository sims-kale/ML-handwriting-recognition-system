

from preprocessing import process_image, process_video
from feature_extrcation import feature_extraction

def main():
    dataset_path = r'D:\SHU\ML_lab\Assesment\Number_Test_Data'
    feature_extraction(process_image)
    # process_image(dataset_path)

    # process_video(dataset_path)

if __name__ == "__main__":
    main()
