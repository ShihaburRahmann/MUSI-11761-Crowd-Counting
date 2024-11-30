import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import preprocessing

# image_paths = glob.glob("raw_images/*.jpg")
# background_path = "raw_images/background/1660366800.jpg"

def count_people_on_beach(image_paths, background_path):
    """
    Count people on beach images by comparing with an empty beach background
    
    Args:
        image_paths (list): List of paths to input images
        background_path (str): Path to empty beach image
    """
    # Read background image
    background = cv2.imread(background_path)
    if background is None:
        raise ValueError(f"Could not read background image: {background_path}")
    
    # Convert background to grayscale
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    # Process each image
    for img_path in image_paths:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
            
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ensure images have same dimensions
        if img_gray.shape != background_gray.shape:
            print(f"Image {img_path} has different dimensions than background")
            continue
            
        # Subtract background
        diff = cv2.absdiff(img_gray, background_gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Apply threshold to create binary image
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to remove noise and connect components
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area to remove small noise
        min_area = 50  # Adjust this value based on your image scale
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Draw contours and count people
        result_img = img.copy()
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate average person size and estimate count
        if filtered_contours:
            areas = [cv2.contourArea(cnt) for cnt in filtered_contours]
            avg_person_area = np.mean(areas)
            estimated_count = len(filtered_contours)
            
            # Adjust count based on potential groups
            large_groups = sum(1 for area in areas if area > 2.5 * avg_person_area)
            estimated_count += large_groups  # Add extra person for each large group
        else:
            estimated_count = 0
        
        # Display results
        plt.figure(figsize=(20, 5))
        
        plt.subplot(141)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(142)
        plt.title('Background')
        plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(143)
        plt.title('Difference Detection')
        plt.imshow(thresh, cmap='gray')
        plt.axis('off')
        
        plt.subplot(144)
        plt.title(f'Detected People (Est. Count: {estimated_count})')
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Estimated number of people in {img_path}: {estimated_count}")
        
        # Return detailed analysis
        analysis = {
            'image_path': img_path,
            'estimated_count': estimated_count,
            'num_contours': len(filtered_contours),
            'avg_person_area': avg_person_area if filtered_contours else 0,
            'detected_areas': areas if filtered_contours else []
        }
        
        return analysis

def analyze_multiple_frames(image_paths, background_path):
    """
    Analyze multiple frames and compute average statistics
    
    Args:
        image_paths (list): List of paths to input images
        background_path (str): Path to empty beach image
    """
    all_analyses = []
    
    for path in image_paths:
        analysis = count_people_on_beach([path], background_path)
        if analysis:
            all_analyses.append(analysis)
    
    if all_analyses:
        avg_count = np.mean([a['estimated_count'] for a in all_analyses])
        print(f"\nAverage number of people across all frames: {avg_count:.1f}")
        
        # Time series analysis if timestamps are available
        counts_over_time = [a['estimated_count'] for a in all_analyses]
        plt.figure(figsize=(10, 5))
        plt.plot(counts_over_time)
        plt.title('People Count Over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Estimated Count')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Get list of image paths (assuming they're in JPG format)
    image_paths = glob.glob("raw_images/*.jpg")
    background_path = "raw_images/background/1660366800.jpg"
    
    # Analyze single or multiple frames
    analyze_multiple_frames(image_paths, background_path)