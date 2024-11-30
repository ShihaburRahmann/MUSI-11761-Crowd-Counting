import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from params import *

from utils import (
    ROI_mask,
    load_labels,
    non_maximum_suppression_proximity,
    calculate_metrics,
    analyze_metrics
)

def count_people_on_beach(image_path, background_path, csv_path):
    """
    Detect and count people on a beach using background subtraction and contour detection.
    """
    # Read and process background image
    background = cv2.imread(background_path)
    ROI_background = ROI_mask(background_path)
    if background is None:
        raise ValueError(f"Could not read background image: {background_path}")
    
    # Read and process input image
    img = cv2.imread(image_path)
    ROI_img = ROI_mask(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert images to grayscale for processing
    background_gray = cv2.cvtColor(ROI_background, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(ROI_img, cv2.COLOR_BGR2GRAY)
    
    if img_gray.shape != background_gray.shape:
        raise ValueError("Image and background must have the same dimensions")
    
    # Image processing pipeline
    diff = cv2.absdiff(img_gray, background_gray)
    blurred = cv2.GaussianBlur(diff, GAUSSIAN_BLUR_KERNEL, 0)

    # Edge detection using Sobel operators
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL_SIZE)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    blurred = np.uint8(np.clip(sobel, 0, 255))

    # Binary thresholding and morphological operations
    _, thresh = cv2.threshold(blurred, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Contour detection and filtering
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < MAX_CONTOUR_WIDTH:
                filtered_contours.append((x, y, w, h))

    # Apply NMS to remove overlapping detections
    non_overlapping_contours = non_maximum_suppression_proximity(
        filtered_contours, 
        distance_threshold=NMS_DISTANCE_THRESHOLD, 
        iou_threshold=NMS_IOU_THRESHOLD
    )

    # Visualization and results processing
    result_img = img.copy()
    centers = []

    # Calculate centroids and draw detection points
    for x, y, w, h in non_overlapping_contours:
        center_x = x + w // 2
        center_y = y + h // 2
        centers.append([center_x, center_y])
        cv2.circle(result_img, (center_x, center_y), 
                  DETECTION_CIRCLE_RADIUS, DETECTION_COLOR, -1)

    # Load ground truth and calculate metrics
    estimated_count = len(non_overlapping_contours)
    image_name = image_path.split("/")[-1]
    original_points = load_labels(csv_path, image_name)
    precision, recall, f1_score, rmse = calculate_metrics(centers, original_points, VAL_DISTANCE_THRESHOLD)
    
    # Append metrics to CSV
    os.makedirs(os.path.dirname(METRICS_OUTPUT_PATH), exist_ok=True)
    
    # Prepare metrics data
    metrics_data = {
        'image_name': image_name,
        'detected_count': estimated_count,
        'ground_truth_count': len(original_points),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score, 4),
        'rmse': round(rmse, 4)
    }
    
    # Check if file exists to determine if header is needed
    file_exists = os.path.isfile(METRICS_OUTPUT_PATH)
    
    # Append to CSV
    with open(METRICS_OUTPUT_PATH, mode='a', newline='') as file:
        fieldnames = ['image_name', 'detected_count', 'ground_truth_count',
                     'precision', 'recall', 'f1_score', 'rmse']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_data)
    
    print(f"\nMetrics appended to {METRICS_OUTPUT_PATH}")

    # Print performance metrics
    print("Precision", round(precision, 4))
    print("Recall", round(recall, 4))
    print("F1 Score", round(f1_score, 4))
    print("RMSE", round(rmse, 4))

    # Visualization of ground truth
    original_img = img.copy()
    for x, y in original_points:
        cv2.circle(original_img, (int(x), int(y)), 
                  DETECTION_CIRCLE_RADIUS, GROUND_TRUTH_COLOR, -1)

    # Display results
    plt.figure(figsize=PLOT_FIGURE_SIZE)
    
    plt.figtext(0.5, 0.98, f"Image: {image_name}", ha="center", va="top", fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'))

    # Create subplots
    plt.subplot(131)
    plt.title('Processed Binary Image')
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.title(f'Detected People (Est. Count: {estimated_count})')
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(133)
    plt.title(f'Original Labelled People (Est. Count: {len(original_points)})')
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

    print(f"Original labeled points for {image_name}: {len(original_points)}")
    
    return {
        'image_path': image_path,
        'estimated_count': estimated_count,
        'num_contours': len(non_overlapping_contours),
    }

def main():
    """
    Main execution function for crowd counting system.
    """
    if os.path.exists(METRICS_OUTPUT_PATH):
        os.remove(METRICS_OUTPUT_PATH)
        print(f"Deleted existing metrics file: {METRICS_OUTPUT_PATH}")

    for filename in os.listdir(IMAGES_FOLDER_PATH):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = os.path.join(IMAGES_FOLDER_PATH, filename).replace("\\", "/")
            try:
                analysis = count_people_on_beach(img, BACKGROUND_IMAGE_PATH, GROUND_TRUTH_PATH)
                print(f"Final estimated count: {analysis['estimated_count']}")
            except Exception as e:
                print(f"Error processing images: {str(e)}")
        
    analyze_metrics(METRICS_OUTPUT_PATH)

if __name__ == "__main__":
    main()