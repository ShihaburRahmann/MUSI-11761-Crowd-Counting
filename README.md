
# Crowd Counting System
## Authors:
- Aina T
- Shihabur Rahman Samrat

A computer vision system for counting people in beach images using traditional methods instead of Machine Learning or Deep Learning. This project is part of our course "Image and Video Analysis" for the MUSI master's programme at UIB.

## Installation

1. Install the required packages:
```
pip install -r requirements.txt
```

## Project Structure

- `dataset/`: Contains the raw images, background image, as well as the annotated ground truth.
- `code/`: Contains all the necessary code to run the algorithm.
   - `params.py`: Contains all configurable parameters (thresholds, paths, etc.)
   - `utils.py`: Contains utility functions (IoU calculation, metrics, etc.)
   - `crowd_counting.py`: Main script for crowd detection and counting.
- `results/`: Contains the CSV file with all the metrics of each image after running the code.
- `code_testing/`: Contains files used for testing by us before developing the main algorithm. This can be ignored.

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