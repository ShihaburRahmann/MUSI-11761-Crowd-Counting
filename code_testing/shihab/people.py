import numpy as np
import cv2

# Load the input image
image = cv2.imread('raw_images/1660377600.jpg')

## Get the dimensions of the image
height, width, _ = image.shape

# Define a rectangular ROI (top to 420 pixels below, full width)
# Define the region of interest (ROI) from the top to 420 pixels below (covering the full width)
roi_corners = np.array([[(0, 900), (500, height), (width, height), (width, 400), (0, 510)]], dtype=np.int32)

# Create a blank mask with the same dimensions as the input image (single channel)
mask = np.zeros_like(image[:, :, 0])  # This is a grayscale mask

# Fill the polygon area (ROI) with white (255)
cv2.fillPoly(mask, roi_corners, 255)
# mask = cv2.bitwise_not(mask)

# Apply the mask to the input image (or any other image you want to process)
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Show the masked image
cv2.imshow("Masked Image", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
