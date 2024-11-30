import cv2
import numpy as np
from matplotlib import pyplot as plt
import os 

def histogram_count_values(image: np.ndarray, nbins: int) -> np.ndarray:
    """Creates a histogram of a grayscale image."""
    size_x = image.shape[0]
    size_y = image.shape[1]
    hist = np.zeros(nbins) 
    for i in range(size_x):
        for j in range(size_y):
            value = image[i, j]
            discretized_value = int(value * (nbins-1) / 255)
            hist[discretized_value] += 1
    return hist

def windowing(image: np.ndarray, lower_threshold: float, upper_threshold: float) -> np.ndarray:
    """Linear normalization assigning values lower or equal to lower_threshold to 0, and values greater or equal to upper_threshold to 255."""
    out = (image - lower_threshold) / (upper_threshold - lower_threshold)
    out[out < 0] = 0
    out[out > 1] = 1
    return out*255


def minmax_normalization(image: np.ndarray) -> np.ndarray:
    """Linear normalization assigning the lowest value to 0 and the highest value to 255."""
    return windowing(image, np.min(image), np.max(image))

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Histogram equalization."""
    hist = histogram_count_values(image, nbins=256)
    cumhist = np.cumsum(hist)
    mapping = minmax_normalization(cumhist)
    return mapping[image.astype('uint8')]

def clahe(image: np.ndarray, clip_limit=5.0, grid_size=(4, 4)) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization."""
    image = image.astype('uint8')   
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)

def image_averaging(images, type, results):
    image_data = []
    for i in images:
        image_data.append(results[i][type])
    avg_image = image_data[0]
    for i in range(len(image_data)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)
    return avg_image        
    


def background_substraction(img_gray,background_gray):
    return cv2.absdiff(img_gray, background_gray)

def apply_sobel(image):
    # Apply Sobel operator to emphasize edges
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction
    sobel = cv2.magnitude(sobel_x, sobel_y)  # Combine gradients

    # Normalize Sobel output to 8-bit for visualization
    blurred = np.uint8(np.clip(sobel, 0, 255))
    return blurred

def filterContours(thresh, minArea, MaxArea):
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area, aspect ratio, and maximum width
    min_area = minArea  # Adjust this value based on your image scale
    max_width = MaxArea  # Maximum allowed width for a single person contour
    filtered_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < max_width:
                filtered_contours.append((x, y, w, h))
    non_overlapping_contours = []
    for i, box1 in enumerate(filtered_contours):
        keep = True
        for j, box2 in enumerate(filtered_contours):
            if i != j and iou(box1, box2) > 0.8:  # Adjust IoU threshold as needed
                keep = False
                break
        if keep:
            non_overlapping_contours.append(box1)   
    return non_overlapping_contours                   

# Remove overlapping bounding boxes
def iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate union
    union = (w1 * h1) + (w2 * h2) - intersection
    
    return intersection / union if union > 0 else 0

