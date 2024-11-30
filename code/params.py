# Directories and Folders
METRICS_OUTPUT_PATH = "results/metrics.csv"
BACKGROUND_IMAGE_PATH = "dataset/background_image/1660366800.jpg"
IMAGES_FOLDER_PATH = 'dataset/images'
GROUND_TRUTH_PATH = "dataset/ground_truth/labels.csv"

# Image Processing Thresholds
GAUSSIAN_BLUR_KERNEL = (13, 13)  # Kernel size for Gaussian blur
SOBEL_KERNEL_SIZE = 3           # Kernel size for Sobel operator
BINARY_THRESHOLD = 39           # Threshold value for binary image
MORPH_KERNEL_SIZE = (3, 3)      # Kernel size for morphological operations

# Contour Filtering Parameters
MIN_CONTOUR_AREA = 5            # Minimum area for valid contours
MAX_CONTOUR_WIDTH = 50          # Maximum width for a single person contour

# Non-Maximum Suppression Parameters
NMS_DISTANCE_THRESHOLD = 30      # Minimum distance between detected people
NMS_IOU_THRESHOLD = 0.09        # Maximum allowed IoU between detections

# Validation Thresholds
VAL_DISTANCE_THRESHOLD = 30     # Detected points within this range of original points will be considered successful

# Visualization Parameters
DETECTION_CIRCLE_RADIUS = 5      # Radius of circles drawn for detections
DETECTION_COLOR = (0, 255, 0)    # Green color for detected points
GROUND_TRUTH_COLOR = (255, 0, 0) # Blue color for ground truth points
PLOT_FIGURE_SIZE = (16, 4)      # Size of the output visualization