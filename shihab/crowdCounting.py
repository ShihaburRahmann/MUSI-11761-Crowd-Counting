import cv2
import numpy as np
import matplotlib.pyplot as plt

def count_people_on_beach(image_path, background_path):
    """
    Count people on a beach image by comparing with an empty beach background
    
    Args:
        image_path (str): Path to input image with people
        background_path (str): Path to empty beach image
        
    Returns:
        dict: Analysis results including estimated count and other metrics
    """
    # Read background image
    background = cv2.imread(background_path)
    if background is None:
        raise ValueError(f"Could not read background image: {background_path}")
    
    # Read input image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert both images to grayscale
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to remove noise and connect components
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area, aspect ratio, and maximum width
    min_area = 1  # Adjust this value based on your image scale
    max_width = 100  # Maximum allowed width for a single person contour
    filtered_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < max_width:
                filtered_contours.append((x, y, w, h))
    
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
    
    # Filter contours by IoU
    non_overlapping_contours = []
    for i, box1 in enumerate(filtered_contours):
        keep = True
        for j, box2 in enumerate(filtered_contours):
            if i != j and iou(box1, box2) > 0.8:  # Adjust IoU threshold as needed
                keep = False
                break
        if keep:
            non_overlapping_contours.append(box1)
    
    # Draw non-overlapping contours and count people
    result_img = img.copy()
    for x, y, w, h in non_overlapping_contours:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    estimated_count = len(non_overlapping_contours)
    
    # Display results
    plt.figure(figsize=(20, 10))
    
    plt.subplot(121)
    plt.title('Difference Detection')
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')
    
    plt.subplot(122)
    plt.title(f'Detected People (Est. Count: {estimated_count})')
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Estimated number of people in {image_path}: {estimated_count}")
    
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
    image_path = "raw_images/1660388400.jpg"
    image_path = "raw_images/1660395600.jpg"

    
    try:
        analysis = count_people_on_beach(image_path, background_path)
        print("\nDetailed Analysis:")
        print(f"Number of non-overlapping contours: {analysis['num_contours']}")
        print(f"Final estimated count: {analysis['estimated_count']}")
    except Exception as e:
        print(f"Error processing images: {str(e)}")

if __name__ == "__main__":
    main()
