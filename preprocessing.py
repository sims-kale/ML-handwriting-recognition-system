import cv2
import os
import matplotlib.pyplot as plt

def preprocess_images(dataset_path):

    # IMAGE PRE_PROCESSING

    try:    
        process_images = []  # List to store images

        for image_name in sorted(os.listdir(dataset_path)):
            image_path = os.path.join(dataset_path, image_name)

            if image_name.endswith('.png'):  # Ensure only images are processed not .avi 
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)   # Gray Scale image
                grayscale_image = grayscale_image[:,:,2]
                resized_image = cv2.resize(grayscale_image, (255, 255))  # Resize image
                blurred_image = cv2.GaussianBlur(resized_image, (1, 1), 0)   #gaussian Blur
                _, binarized_image = cv2.threshold(blurred_image, 80, 220, cv2.THRESH_BINARY_INV)
                process_images.append(binarized_image)

        # Display images in 3x3 grid using subplots
        f, axes = plt.subplots(3, 5, figsize=(7, 7))

        for i, ax in enumerate(axes.flatten()):
            ax.imshow(process_images[i], cmap='gray')
            ax.axis('off')

        plt.show()

        return process_images

    except Exception as e:
        print(f"Error occurred in image preprocessing: {str(e)}")

    

def process_video(dataset_path):
    

    # ".AVI" VIDEO PRE_PROCESSING

    try:
        for video_path in (os.listdir(dataset_path)):
            
            video = os.path.join(dataset_path, video_path)
            video_op_path = os.path.join(dataset_path, 'video_frames')
            if not os.path.exists(video_op_path):
                os.makedirs(video_op_path)
                
            
            
            if video_path.endswith('.avi'):
                # print(video)
                videoCap = cv2.VideoCapture(video)      #Read the video
        
                frameCount = 0
                while videoCap.isOpened():
                    reading, frame = videoCap.read()
                    if not reading:
                        break    #stop if the video finish

                    grayscale_frames = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
                    resize_frames = cv2.resize(grayscale_frames, (255, 255))

                    # Apply Gaussian Blur

                    blurred_frame = cv2.GaussianBlur(resize_frames, (1, 1), 0)
                    _, binarized_frame = cv2.threshold(blurred_frame, 170, 255, cv2.THRESH_BINARY_INV)
                    cv2.imshow("Processed Frame", binarized_frame)
                    filename = f"frame_{frameCount}.png"
                    path = os.path.join(video_op_path, filename)
                    if frameCount %10 == 0:
                        cv2.imwrite(path, binarized_frame)
                    # print(video_op_path,filename)
                    frameCount += 1
                    cv2.waitKey(10)
                
                videoCap.release()
                # cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error occurred in image preprocessing: {str(e)}")


