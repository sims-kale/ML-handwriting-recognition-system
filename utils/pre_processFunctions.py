import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


# Load first 10 images
def zero_nine_images(image, **kwargs):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # resize_digit = cv2.resize(image, (28, 28), interpolation= cv2.INTER_AREA) 
    # np_digit = resize_digit.astype(np.float32) / 255.0
    # blur_digit = cv2.GaussianBlur(np_digit, (3, 3), 0)
    # image = np.expand_dims(blur_digit, axis=-1) 
    if image is None:
        raise ValueError(f"Unable to load image: {image}")
    print(f"Loaded image shape: {image.shape}")  # Print the shape of the raw image
    return image  # Return the raw image
# Load remaining images
def load_images(image):
    image = cv2.imread(image)
    return image

# Load, resize and grayscale the images
def resized_images(image):
    resized_image = cv2.resize(image, (580, 580), interpolation=cv2.INTER_AREA)

    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Image", gray_image)
    # cv2.waitKey(0)

    return gray_image

# Apply CLAHE for contrast enhancement
def enhance_contrast(image, clip_limit=2.0):
    
    clahe = cv2.createCLAHE(clip_limit, tileGridSize=(11, 11))
    enhanced_image = clahe.apply(image)
    # cv2.imshow("Enhanced Image", enhanced_image)
    # cv2.waitKey(0)
    return enhanced_image

# Apply either binary or adaptive thresholding.
def apply_threshold(image, method, threshold):
    
    print(f"Applying thresholding method: {method}, threshold: {threshold}")
    if method == "13_binary":
        _, bin_img = cv2.threshold(image, threshold, 225, cv2.THRESH_BINARY_INV)
    elif method == "adaptive":
        bin_img = cv2.GaussianBlur(image, (5,5), 0)
        bin_img = cv2.adaptiveThreshold(
            bin_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
        )
    else:
        bin_img = image

    cv2.imshow("Thresholded Image", bin_img)
    cv2.waitKey(0)
    return bin_img


# Detect and remove lines using Hough Line Transform
def remove_lines(thresholded_image, method):

    if method == "houghLine":
        lines = cv2.HoughLinesP(
            thresholded_image, 1, np.pi / 180, 100, minLineLength=5, maxLineGap=90
        )
        mask = np.zeros_like(thresholded_image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(mask, (x1, y1), (x2, y2), (255), 2)
        thresholded_image = cv2.inpaint(thresholded_image, mask, 3, cv2.INPAINT_TELEA)

    elif method == "kernelLine":
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1 ))
        detected_lines = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, line_kernel, iterations=1)
        cnts_lines, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_lines:
            cv2.drawContours(thresholded_image, [c], -1, (0) -1)

    # cv2.imshow("Lines Removed", thresholded_image)
    # cv2.waitKey(0)
    return thresholded_image

# Optional: Apply a slight erosion to separate merged digits
def apply_erosion_dilation(thresholded_image):
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresholded_image = cv2.erode(thresholded_image, kernel_small, iterations=1)
    cv2.imshow("Eroded Image", thresholded_image)
    cv2.waitKey(0)
    return thresholded_image

# extract digits from images
def extract_digits(thresholded_image, area):
    # Debug: Visualize contours
    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # debug_image = cv2.cvtColor(thresholded_image.copy(), cv2.COLOR_GRAY2BGR)
   
    for c in contours:
        
        # Filter out too-small or too-large bounding boxes
        areas = cv2.contourArea(c)        
        if areas < area:
            cv2.drawContours(thresholded_image, [c], -1, (0), -1)
        countors = [c for c in contours if cv2.contourArea(c) > area] 
        # Sort by y first (for multiple rows) then x (for left-to-right)
        countors.sort(key=lambda x: cv2.boundingRect(x)[0])
        # countors.sort(key=lambda x: (cv2.boundingRect(x)[1], cv2.boundingRect(x)[0]))
        
        # bounding_boxes = sorted(
        #     bounding_boxes, 
        #     key=lambda b: (round((b[1] + b[3] / 2) / 30), b[0] + b[2] / 2)
        # )
    
    # Draw green bounding boxes for visualization
    debug_image = cv2.cvtColor(thresholded_image.copy(), cv2.COLOR_GRAY2BGR)
    for cnt in countors:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Detected Green Boxes", debug_image)
    cv2.waitKey(0)

    digits = []
    for cnt in countors:
        x, y, w, h = cv2.boundingRect(cnt)
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(thresholded_image.shape[1], x + w + padding)
        y_end = min(thresholded_image.shape[0], y + h + padding)

        digit = thresholded_image[y_start:y_end, x_start:x_end]
        digit = cv2.resize(digit, (28, 28), interpolation= cv2.INTER_AREA) # Resize to 28x28
        # digit = digit.astype(np.float32) / 255.0 # Normalize to 0-1 range
        # digit = cv2.GaussianBlur(digit, (3, 3), 0)  #gaussian blur
        # digit = np.expand_dims(digit, axis=-1)  # Add channel dimension

        digits.append(digit)

    plt.figure(figsize=(10, 2))
    for i, digit_image in enumerate(digits):
        plt.subplot(1, len(digits), i + 1)
        plt.imshow(digit_image, cmap="gray")
        plt.axis("on")
    plt.show()
    return digits

# Run a series of processing steps defined by the pipeline on an image
def process_image(image, pipeline):
    
    try:
        print(f"Processing image: {image}")
        
        if image is None:
            raise ValueError(f"Unable to load image: {image}")
        
        for step in pipeline:
            func = step["func"]
            kwargs = step.get("kwargs", {})
            print(f"Running step: {func.__name__} with kwargs: {kwargs}")
            image = func(image, **kwargs)
        
        return image
    except Exception as e:
        print(f"Error processing image {image}: {e}")

# Process the image with the given pipeline and extract digit regions
def process_and_extract_digits(image, pipeline):
    
    try:
        for step in pipeline:
            func = step["func"]
            kwargs = step.get("kwargs", {})
            print(f"Running step: {func.__name__} with kwargs: {kwargs}")
            image = func(image, **kwargs)
        return image
    except Exception as e:
        print(f"Error processing image {image}: {e}")
        return None
    
def detect_lines(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = 20
    countors = [c for c in contours if cv2.contourArea(c) > area]

    # Sort contours from top to bottom, left to right
    
    row_size = image.shape[1]/ 5 
    sort_form = (y // row_size) * image.shape[1] + x
    countors = sorted(sort_form)
    # Draw green bounding boxes for visualization
    debug_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    for cnt in countors:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Detected Green Boxes", debug_image)
    cv2.waitKey(0)

    digits = []
    for cnt in countors:
        x, y, w, h = cv2.boundingRect(cnt)
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)

        digit = image[y_start:y_end, x_start:x_end]
        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)  # Resize to 28x28
        digits.append(digit)

    plt.figure(figsize=(10, 2))
    for i, digit_image in enumerate(digits):
        plt.subplot(1, len(digits), i + 1)
        plt.imshow(digit_image, cmap="gray")
        plt.axis("on")
    plt.show()
    return digits

# Define custom pipelines (using your file structure and code logic)
custom_processes= {
    "image_0_9": [
        {"func": zero_nine_images, "kwargs": {}},  # Ensure this points to a valid function
    ],
    "image_10":[
        {"func": load_images, "kwargs":{}},
        {"func": detect_lines, "kwargs":{}},
    ],
    "image_13": [
        {"func": load_images, "kwargs": {}},
        {"func": resized_images, "kwargs": {}},
        # {"func": enhance_contrast, "kwargs": {"clip_limit": 2.0}},
        {"func": apply_threshold, "kwargs": {"method": "13_binary","threshold" : 85}},
        {"func": remove_lines, "kwargs": {"method": "houghLine"}},
        {"func": apply_erosion_dilation, "kwargs": {}},
        {"func": extract_digits, "kwargs": {"area": 30}},
        
    ],
    "image_14": [
        {"func": load_images, "kwargs": {}},
        {"func": resized_images, "kwargs": {}},
        {"func": apply_threshold, "kwargs": {"method": "adaptive", "threshold" : 40}},
        {"func": remove_lines, "kwargs": {"method": "houghLine"}},
        {"func": apply_erosion_dilation, "kwargs": {}},
        {"func": extract_digits, "kwargs": {"area": 70}},

        # {"func": extract_digits, "kwargs": {}},  # Assuming this function returns the digits
        
    ],
}


def process_video(dataset_path):

    # ".AVI" VIDEO PRE_PROCESSING

    try:
        for video_path in os.listdir(dataset_path):

            video = os.path.join(dataset_path, video_path)
            video_op_path = os.path.join(dataset_path, "video_frames")
            if not os.path.exists(video_op_path):
                os.makedirs(video_op_path)

            if video_path.endswith(".avi"):
                # print(video)
                videoCap = cv2.VideoCapture(video)  # Read the video

                frameCount = 0
                while videoCap.isOpened():
                    reading, frame = videoCap.read()
                    if not reading:
                        break  # stop if the video finish

                    grayscale_frames = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    resize_frames = cv2.resize(grayscale_frames, (255, 255))

                    # Apply Gaussian Blur

                    blurred_frame = cv2.GaussianBlur(resize_frames, (1, 1), 0)
                    _, binarized_frame = cv2.threshold(
                        blurred_frame, 170, 255, cv2.THRESH_BINARY_INV
                    )
                    filename = f"frame_{frameCount}.png"
                    path = os.path.join(video_op_path, filename)
                    if frameCount % 10 == 0:
                        cv2.imwrite(path, binarized_frame)
                    # print(video_op_path,filename)
                    frameCount += 1
                    cv2.waitKey(10)

                videoCap.release()
                # cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error occurred in image preprocessing: {str(e)}")
