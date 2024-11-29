import numpy as np
import cv2
from scipy.spatial import distance
import pandas as pd

def compute_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def compute_center(box):
    """
    Calculate the center coordinates of a bounding box.
    """
    x, y, w, h = box
    return (x + w / 2, y + h / 2)

def compute_distance(center1, center2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.linalg.norm(np.array(center1) - np.array(center2))

def non_maximum_suppression_proximity(boxes, distance_threshold=50, iou_threshold=0.3):
    """
    Perform Non-Maximum Suppression considering both distance and IoU.
    """
    selected_boxes = []
    remaining_boxes = boxes.copy()
    
    while remaining_boxes:
        current_box = remaining_boxes.pop(0)
        selected_boxes.append(current_box)
        
        current_center = compute_center(current_box)
        remaining_boxes = [box for box in remaining_boxes if 
                           compute_distance(current_center, compute_center(box)) > distance_threshold and 
                           compute_iou(current_box, box) < iou_threshold]
    
    return selected_boxes

def load_labels(csv_path, image_name):
    """
    Load annotation points from CSV file.
    """
    try:
        labels = pd.read_csv(csv_path, header=None, names=["Label", "X", "Y", "Image", "Width", "Height"])
        points = labels[labels["Image"] == image_name][["X", "Y"]].values.tolist()
        return points
    except Exception as e:
        print(f"Error reading labels: {e}")
        return []

def ROI_mask(image_path):
    """
    Create a Region of Interest mask.
    """
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    roi_corners = np.array([[(0, 900), (500, height), (width, height), (width, 400), (0, 510)]], dtype=np.int32)
    mask = np.zeros_like(image[:, :, 0])
    cv2.fillPoly(mask, roi_corners, 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def calculate_distance(centroid, annotation):
    """
    Calculate distances between centroids and annotations.
    """
    return distance.cdist(centroid, annotation, 'euclidean')

def calculate_mse(tp, fp, len):
    """
    Calculate Mean Squared Error.
    """
    return ((tp + fp) - len)**2

def calculate_metrics(centroids, annotations, matching_threshold=30):
    """
    Calculate performance metrics.
    """
    tp = 0
    fp = 0
    distances = []
    match_annotations = np.zeros(len(annotations))
    distances = calculate_distance(np.array(centroids), np.array(annotations))
    
    for i, c in enumerate(centroids):
        min = np.min(distances[i,:])
        closest_annotation = np.argmin(distances[i,:])
        if min <= matching_threshold and not match_annotations[closest_annotation]:
            tp += 1
            match_annotations[closest_annotation] = 1
        else:
            fp += 1
            
    total_detected = np.count_nonzero(match_annotations)  
    fn = len(match_annotations) - total_detected     

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)  
    f1_score = (2 * precision * recall) / (precision + recall)
    mse = calculate_mse(tp, fp, len(annotations))
    rmse = np.sqrt(mse) 

    return precision, recall, f1_score, rmse