import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


# Load, resize and grayscale the images
def load_resized_images(image):
    resized_image = cv2.resize(image, (580, 580), interpolation=cv2.INTER_AREA)

    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image", gray_image)
    cv2.waitKey(0)

    return gray_image


def enhance_contrast(image, clip_limit=2.0):
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clip_limit, tileGridSize=(10, 10))
    enhanced_image = clahe.apply(image)
    cv2.imshow("Enhanced Image", enhanced_image)
    cv2.waitKey(0)
    return enhanced_image


def apply_threshold(image, method="14_binary"):
    """Apply either binary or adaptive thresholding."""
    if method == "14_binary":
        _, bin_img = cv2.threshold(image, 85, 225, cv2.THRESH_BINARY_INV)
    elif method == "adaptive":
        bin_img = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
        )
    else:
        bin_img = image
    return bin_img


# Detect and remove lines using Hough Line Transform
def remove_lines(thresholded_image, apply_line_removal=True):
    if not apply_line_removal:
        return thresholded_image
    lines = cv2.HoughLinesP(
        thresholded_image, 1, np.pi / 180, 100, minLineLength=30, maxLineGap=19.5
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(thresholded_image, (x1, y1), (x2, y2), (0, 0, 0), 2)
    cv2.imshow("Lines Removed", thresholded_image)
    cv2.waitKey(0)
    # line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1 ))
    # detected_lines = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, line_kernel, iterations=1)
    # cnts_lines, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for c in cnts_lines:
    #     cv2.drawContours(thresholded_image, [c], -1, (0, 0, 0), thickness=cv2.FILLED)
    return thresholded_image


def apply_erosion_dilation(thresholded_image):
    # Optional: Apply a slight erosion to separate merged digits
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresholded_image = cv2.erode(thresholded_image, kernel_small, iterations=1)
    cv2.imshow("Eroded Image", thresholded_image)
    cv2.waitKey(0)
    return thresholded_image


# extract digits from images
def extract_digits(erosion_image):
    # Debug: Visualize contours
    contours, _ = cv2.findContours(
        erosion_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    debug_image = cv2.cvtColor(erosion_image.copy(), cv2.COLOR_GRAY2BGR)
    bounding_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # area = cv2.contourArea(c)
        # Filter out too-small or too-large bounding boxes
        if w < 3 or h < 10 or w > 200 or h > 200:
            continue
        bounding_boxes.append((x, y, w, h))
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Bounding Boxes", debug_image)
    cv2.waitKey(0)

    # Sort by y first (for multiple rows) then x (for left-to-right)
    bounding_boxes = sorted(
        bounding_boxes, 
        key=lambda b: (round((b[1] + b[3] / 2) / 30), b[0] + b[2] / 2)
    )

    digits = []
    for x, y, w, h in bounding_boxes:
        digit_roi = erosion_image[y : y + h, x : x + w]
        digits_roi = cv2.resize(digit_roi, (28, 28)) # Resize to 28x28
        digit_roi = digits_roi.astype(np.float32) / 255.0 # Normalize to 0-1 range
        # digit_roi = cv2.GaussianBlur(digit_roi, (3, 3), 0)  #gaussian blur

        digits.append(digit_roi)

    # Display digits for this image
    for digit_img in digits:
        cv2.imshow("Digit", digit_img)
        cv2.waitKey(0)

    plt.figure(figsize=(10, 2))
    for i, digit_image in enumerate(digits):
        plt.subplot(1, len(digits), i + 1)
        plt.imshow(digit_image, cmap="gray")
        plt.axis("off")
    plt.show()
    return digits


# def make_mnist_size(image, size=28):
#     h, w = image.shape
#     if h > w:
#         new_h = 20
#         new_w = int(round(w * 20.0 / h))
#     else:
#         new_w = 20
#         new_h = int(round(h * 20.0 / w))
#     resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
#     canvas = np.zeros((28, 28), dtype=np.uint8)
#     x_offset = (28 - new_w) // 2
#     y_offset = (28 - new_h) // 2
#     canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_image
#     canvas = shift_to_center_of_mass(canvas)
#     return canvas


# def shift_to_center_of_mass(image):
#     m = cv2.moments(image)
#     if m["m00"] == 0:
#         return image
#     c_x = int(m["m10"] / m["m00"])
#     c_y = int(m["m01"] / m["m00"])
#     shift_x = 14 - c_x
#     shift_y = 14 - c_y
#     M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
#     shifted = cv2.warpAffine(image, M, (28, 28), borderValue=0)
#     return shifted


# def preprocess_images(image_paths):
#     all_digits = []  # List to store digits for each image
#     for image in image_paths:
#         # Load, resize and grayscale the images
#         gray_image = load_resized_images(image)
#         # Apply contrast enhancement
#         enhanced_image = enhance_contrast(gray_image)
#         # Apply thresholding
#         thresholded_image = apply_threshold(enhanced_image, method="14_binary")
#         cv2.imshow("Thresholded Image", thresholded_image)
#         cv2.waitKey(0)
#         # Remove lines using Hough Line Transform
#         thresholded_image = remove_lines(thresholded_image)
#         erosion_image = apply_erosion_dilation(thresholded_image)
#         # Extract digits from the image
#         digit_images = extract_digits(erosion_image)
#         all_digits.append(digit_images)

#     return all_digits

def process_image(image, pipeline):
    """Run a series of processing steps defined by the pipeline on an image."""
    try:
        
        print(f"Processing image: {image}")
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Unable to load image: {image}")
        for step in pipeline:
            func = step["func"]
            kwargs = step.get("kwargs", {})
            image = func(image, **kwargs)
        return image
    except Exception as e:
        print(f"Error processing image {image}: {e}")
        return None

def process_and_extract_digits(image_path, pipeline):
    """Process the image with the given pipeline and extract digit regions."""
    for image in image_path:
        processed_image = process_image(image, pipeline)
        if processed_image is not None:
            # Optionally display the processed image
            plt.figure(figsize=(6,6))
            plt.imshow(processed_image, cmap="gray")
            plt.axis("off")
            # plt.title(f"Processed: {os.path.basename(image_path)}")
            plt.show()
            digits = extract_digits(processed_image)
            return digits
        else:
            return None

# Define custom pipelines (using your file structure and code logic)
custom_processes= {
    "default": [
        {"func": load_resized_images, "kwargs": {}},
        {"func": enhance_contrast, "kwargs": {"clip_limit": 2.0}},
        {"func": apply_threshold, "kwargs": {"method": "adaptive"}},
        {"func": remove_lines, "kwargs": {"apply_line_removal": False}},
    ],
    "image_14": [
        {"func": load_resized_images, "kwargs": {}},
        {"func": enhance_contrast, "kwargs": {"clip_limit": 2.0}},
        {"func": apply_threshold, "kwargs": {"method": "14_binary"}},
        {"func": remove_lines, "kwargs": {"apply_line_removal": True}},
        {"func": apply_erosion_dilation, "kwargs": {}},

        {"func": extract_digits, "kwargs": {}},  # Assuming this function returns the digits
        
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
