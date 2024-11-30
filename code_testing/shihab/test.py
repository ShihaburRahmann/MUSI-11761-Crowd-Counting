import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

# Load an image
img = mpimg.imread('raw_images/1660388400.jpg')  # Replace with your image file path

# Coordinates for the bounding box (x, y, width, height)
x, y, width, height = 883, 930, 65, 135  # Example values, adjust as needed

# Create a plot
fig, ax = plt.subplots()

# Display the image
ax.imshow(img)

# Create a Rectangle patch and add it to the plot
rect = Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)

# Add text showing the width and height
ax.text(x, y-10, f'W: {width}, H: {height}', color='red', fontsize=12, ha='left')

# Hide axes for cleaner display
ax.axis('off')

# Show the image with the bounding box
plt.show()
