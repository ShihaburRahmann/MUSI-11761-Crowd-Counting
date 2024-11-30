import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def pre_process_image(image):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # 2. Apply Gaussian Blur (optional)
    # # Comment out this line if blurring is not required
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # # 3. Apply Thresholding (optional)
    # # Comment out this line if thresholding is not required
    # _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return gray


def apply_edge_filters_from_folder(folder_path):
    """
    Applies Laplacian, Sobel (x and y), and Prewitt (x and y) filters to all images in a folder.
    Displays the results using matplotlib.

    Args:
        folder_path (str): Path to the folder containing image files.
    """
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return
    
    # Get list of image files in the folder
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print("No image files found in the folder.")
        return
    
    print(f"Processing {len(image_files)} image(s) from the folder: {folder_path}")
    
    for img_path in image_files:
        # Read image in grayscale
        img = cv2.imread(img_path)
        img = pre_process_image(img)
        if img is None:
            print(f"Could not load image: {img_path}")
            continue

        # Apply Laplacian filter
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)

        # Apply Sobel filters
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)

        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = cv2.convertScaleAbs(sobel_y)

        sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

        # Apply Prewitt filters
        prewitt_kernel_x = np.array([[-1, 0, 1], 
                                     [-1, 0, 1], 
                                     [-1, 0, 1]])
        prewitt_kernel_y = np.array([[-1, -1, -1], 
                                     [ 0,  0,  0], 
                                     [ 1,  1,  1]])

        prewitt_x = cv2.filter2D(img, -1, prewitt_kernel_x)
        prewitt_y = cv2.filter2D(img, -1, prewitt_kernel_y)
        prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

        # Display results using matplotlib
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray'), plt.title("Original Image")
        plt.axis("off")
        plt.subplot(2, 3, 2), plt.imshow(laplacian, cmap='gray'), plt.title("Laplacian")
        plt.axis("off")
        plt.subplot(2, 3, 3), plt.imshow(sobel_combined, cmap='gray'), plt.title("Sobel (Combined)")
        plt.axis("off")
        plt.subplot(2, 3, 4), plt.imshow(prewitt_x, cmap='gray'), plt.title("Prewitt (X)")
        plt.axis("off")
        plt.subplot(2, 3, 5), plt.imshow(prewitt_y, cmap='gray'), plt.title("Prewitt (Y)")
        plt.axis("off")
        plt.subplot(2, 3, 6), plt.imshow(prewitt_combined, cmap='gray'), plt.title("Prewitt (Combined)")
        plt.axis("off")
        plt.suptitle(f"Filters Applied: {os.path.basename(img_path)}", fontsize=16)
        plt.tight_layout()
        plt.show()

folder_path = "raw_images/"
apply_edge_filters_from_folder(folder_path)
