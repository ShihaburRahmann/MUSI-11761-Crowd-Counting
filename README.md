
# Crowd Counting System
## Authors:
- Aina T
- Shihabur Rahman Samrat

A computer vision system for counting people in beach images using background subtraction and contour detection.

## Installation

1. Install the required packages:
```
pip install -r requirements.txt
```

## Project Structure

- `params.py`: Contains all configurable parameters (thresholds, paths, etc.)
- `utils.py`: Contains utility functions (IoU calculation, metrics, etc.)
- `crowd_counting.py`: Main script for crowd detection and counting

## Usage

1. Review and modify parameters in `params.py` if needed:
   - Adjust paths to your image files
   - Modify detection thresholds
   - Change visualization settings

2. Run the main script:
```
python crowd_counting.py
```

## Output

- Displays visualization of detected people vs ground truth
- Saves metrics to CSV file in results folder
- Prints detection statistics to console

## Note

Make sure your image paths in `params.py` are correct before running the script.