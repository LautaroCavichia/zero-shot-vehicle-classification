"""
YOLOv8 detector implementation
"""
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from config import DETECTOR_CONFIGS


class YOLOv12Detector:
    """YOLOv12 detector for object detection"""
    
    def __init__(self, model_path: str = None, conf_threshold: float = None):
        """
        Initialize YOLOv12 detector
        
        Args:
            model_path: Path to the YOLOv12 model weights
            conf_threshold: Confidence threshold for detections
        """
        # Use config values if not provided
        config = DETECTOR_CONFIGS['yolov12']
        self.model_path = model_path or config['model_path']
        self.conf_threshold = conf_threshold or config['conf_threshold']
        self.vehicle_classes = config['vehicle_classes']
        
        # Load model
        self.model = YOLO(self.model_path)
        
        # Device configuration
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"YOLOv12 using device: {self.device}")
        
    def detect(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Detect objects in the image
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Tuple of (list of detections, inference time)
        """
        # Start timing
        start_time = time.time()
        
        # Run inference
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        # Process results
        detections = []
        
        # Extract predictions
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            for i, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get confidence and class
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # Only keep vehicle classes
                if cls in self.vehicle_classes:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls,
                        'area': (x2 - x1) * (y2 - y1),
                    })
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        return detections, inference_time
    
    def find_main_vehicle(self, detections: List[Dict], image_shape: Tuple[int, int, int]) -> Dict:
        """
        Find the main vehicle in the image based on centrality and size
        
        Args:
            detections: List of detections from the model
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Dict containing the main vehicle detection
        """
        if not detections:
            return None
        
        # Get image dimensions
        img_height, img_width = image_shape[0], image_shape[1]
        img_center = (img_width / 2, img_height / 2)
        
        # Calculate centrality score for each detection
        for det in detections:
            bbox = det['bbox']
            # Calculate bbox center
            bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # Calculate distance from image center
            distance = np.sqrt((bbox_center[0] - img_center[0])**2 + 
                               (bbox_center[1] - img_center[1])**2)
            
            # Normalize distance by image diagonal
            img_diagonal = np.sqrt(img_width**2 + img_height**2)
            normalized_distance = distance / img_diagonal
            
            # Calculate centrality score (1 - normalized_distance)
            centrality = 1 - normalized_distance
            
            # Calculate size score (normalized by image area)
            size_score = det['area'] / (img_width * img_height)
            
            # Combined score (balance of centrality and size)
            det['score'] = 0.7 * centrality + 0.3 * size_score
        
        # Sort by combined score (descending)
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        # Return the detection with the highest score
        return detections[0] if detections else None
    
    def crop_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Crop image to the specified bounding box
        
        Args:
            image: Input image (RGB format, numpy array)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Crop image
        cropped = image[y1:y2, x1:x2]
        
        return cropped