import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    IoU measures the overlap between two bounding boxes by dividing the area of 
    intersection by the area of union.
    
    Parameters:
        box1 (tuple): First bounding box in format (x, y, width, height)
                      where (x,y) is the top-left corner
        box2 (tuple): Second bounding box in format (x, y, width, height)
                      where (x,y) is the top-left corner
    
    Returns:
        float: IoU score between 0 and 1, where:
               0 = no overlap
               1 = perfect overlap
    """
    # Extract coordinates and dimensions from both boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Find the coordinates of intersection rectangle
    # Taking maximum of x/y coordinates ensures we get the inner-most points
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    
    # Taking minimum of end points ensures we get the inner-most points
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Calculate intersection area
    # max(0, ...) ensures we don't get negative areas when boxes don't overlap
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate individual box areas
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate IoU
    # Union area = sum of both areas - intersection area
    # (to avoid counting the intersection area twice)
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def compute_center(box):
    # Compute the center (x, y) of the bounding box
    x, y, w, h = box
    return (x + w / 2, y + h / 2)

def compute_distance(center1, center2):
    # Compute Euclidean distance between two centers
    return np.linalg.norm(np.array(center1) - np.array(center2))

def non_maximum_suppression_proximity(boxes, distance_threshold=50, iou_threshold=0.3):
    selected_boxes = []
    remaining_boxes = boxes.copy()
    
    while remaining_boxes:
        # Take the first box (largest by area or first in list)
        current_box = remaining_boxes.pop(0)
        selected_boxes.append(current_box)
        
        current_center = compute_center(current_box)
        
        # Filter out boxes that are either too close or have a high IoU with the current box
        remaining_boxes = [box for box in remaining_boxes if 
                           compute_distance(current_center, compute_center(box)) > distance_threshold and 
                           compute_iou(current_box, box) < iou_threshold]
    
    return selected_boxes


def load_labels(csv_path, image_name):
    try:
        labels = pd.read_csv(csv_path, header=None, names=["Label", "X", "Y", "Image", "Width", "Height"])
        points = labels[labels["Image"] == image_name][["X", "Y"]].values.tolist()
        return points
    except Exception as e:
        print(f"Error reading labels: {e}")
        return []

def ROI_mask(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    roi_corners = np.array([[(0, 900), (500, height), (width, height), (width, 400), (0, 510)]], dtype=np.int32)

    # Create a blank mask with the same dimensions as the input image (single channel)
    mask = np.zeros_like(image[:, :, 0])  # This is a grayscale mask

    # Fill the polygon area (ROI) with white (255)
    cv2.fillPoly(mask, roi_corners, 255)

    # Apply the mask to the input image (or any other image you want to process)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image

def count_people_on_beach(image_path, background_path, csv_path):
    # Read background image
    background = cv2.imread(background_path)
    ROI_background = ROI_mask(background_path)
    if background is None:
        raise ValueError(f"Could not read background image: {background_path}")
    
    # Read input image
    img = cv2.imread(image_path)
    ROI_img = ROI_mask(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert both images to grayscale
    background_gray = cv2.cvtColor(ROI_background, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(ROI_img, cv2.COLOR_BGR2GRAY)
    
    # Ensure images have same dimensions
    if img_gray.shape != background_gray.shape:
        raise ValueError("Image and background must have the same dimensions")
    
    # Subtract background
    diff = cv2.absdiff(img_gray, background_gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(diff, (11, 11), 0)

    # Apply Sobel operator to emphasize edges
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction
    sobel = cv2.magnitude(sobel_x, sobel_y)  # Combine gradients

    # Normalize Sobel output to 8-bit for visualization
    blurred = np.uint8(np.clip(sobel, 0, 255))

    # Apply threshold to create binary image
    _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise and connect components
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area, aspect ratio, and maximum width
    min_area = 20  # Adjust this value based on your image scale
    max_width = 50  # Maximum allowed width for a single person contour
    filtered_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < max_width:
                filtered_contours.append((x, y, w, h))

    non_overlapping_contours = non_maximum_suppression_proximity(filtered_contours, distance_threshold=5, iou_threshold=0.1)

    # Draw non-overlapping contours and count people
    result_img = img.copy()
    # for x, y, w, h in non_overlapping_contours:
    #     cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for x, y, w, h in non_overlapping_contours:
    # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Draw a point (circle) at the center of the bounding box
        cv2.circle(result_img, (center_x, center_y), 5, (0, 255, 0), -1)

    estimated_count = len(non_overlapping_contours)
    image_name = image_path.split("/")[-1]  # Extract image name
    original_points = load_labels(csv_path, image_name)

    # Draw original labeled points on the image
    original_img = img.copy()
    for x, y in original_points:
        cv2.circle(original_img, (int(x), int(y)), 5, (255, 0, 0), -1)  # Draw a blue point


    # Find connected components in the binary mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # Filter connected components based on size (optional, to remove noise)
    min_area = 20  # Minimum area of a connected component
    filtered_components = []
    for i in range(1, num_labels):  # Skip the background (label 0)
        x, y, w, h, area = stats[i]
        if area >= min_area:
            filtered_components.append(centroids[i])  # Use the centroid for visualization

    # filtered_components = non_maximum_suppression_proximity(filtered_components, distance_threshold=5, iou_threshold=0.1)

    # Draw points for the filtered connected components
    result_img1 = img.copy()

    for cx, cy in filtered_components:
        cv2.circle(result_img1, (int(cx), int(cy)), 5, (0, 0, 255), -1)  # Draw red points (radius = 5)


    # # Draw the filtered connected components (bounding boxes or centroids)
    # for x, y, w, h in filtered_components:
    #     cv2.rectangle(result_img1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Or draw points using centroids



    # Visualization of Results
    plt.figure(figsize=(20, 10))

    plt.subplot(141)
    plt.title('Difference Detection')
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')

    plt.subplot(142)
    plt.title(f'Detected People (Est. Count: {estimated_count})')
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(143)
    plt.title('Original Labeled Points')
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(144)
    plt.title(f'Detected People CC (Est. Count: {len(filtered_components)})')
    plt.imshow(cv2.cvtColor(result_img1, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Original labeled points for {image_name}: {len(original_points)}")
    
    # Return detailed analysis
    analysis = {
        'image_path': image_path,
        'estimated_count': estimated_count,
        'num_contours': len(non_overlapping_contours),
    }
    
    return analysis


def main():
    # Example usage
    background_path = "raw_images/background/1660366800.jpg"
    image_path = "raw_images/1660402800.jpg"
    # image_path = "raw_images/1660388400.jpg"
    # image_path = "raw_images/1660392000.jpg"

    csv_path = "labels.csv"


    folder_path = 'raw_images/'
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder_path, filename))
            try:
                analysis = count_people_on_beach(img, background_path, csv_path)
                print("\nDetailed Analysis:")
                print(f"Number of detected contours: {analysis['num_contours']}")
                # print(f"Average person area: {analysis['avg_person_area']:.2f} pixels")
                print(f"Final estimated count: {analysis['estimated_count']}")
            except Exception as e:
                print(f"Error processing images: {str(e)}")
            # Process the image (img) here

    # try:
    #     analysis = count_people_on_beach(image_path, background_path, csv_path)
    #     print("\nDetailed Analysis:")
    #     print(f"Number of detected contours: {analysis['num_contours']}")
    #     # print(f"Average person area: {analysis['avg_person_area']:.2f} pixels")
    #     print(f"Final estimated count: {analysis['estimated_count']}")
    # except Exception as e:
    #     print(f"Error processing images: {str(e)}")

if __name__ == "__main__":
    main()