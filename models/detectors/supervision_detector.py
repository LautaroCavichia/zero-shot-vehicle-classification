"""
Supervision detector implementation
"""
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO

from config import DETECTOR_CONFIGS


class SupervisionDetector:
    """Supervision-based detector for object detection"""
    
    # COCO dataset class IDs for vehicles
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    MODEL_SIZES = {
        "small": "yolov8s.pt",
        "medium": "yolov8m.pt",
        "large": "yolov8l.pt",
    }
    
    def __init__(self, model_size: str = "medium", conf_threshold: float = None, custom_model_path: Optional[str] = None):
        """
        Initialize Supervision detector
        
        Args:
            model_size: Size of the model (small, medium, large)
            conf_threshold: Confidence threshold for detections
            custom_model_path: Path to custom model weights (overrides model_size if provided)
        """
        # Use config values if not provided
        config = DETECTOR_CONFIGS['supervision']
        self._model_size = model_size
        self.conf_threshold = conf_threshold or config['conf_threshold']
        self.vehicle_classes = config['vehicle_classes']
        
        # Set up model path
        if custom_model_path and os.path.exists(custom_model_path):
            self.model_path = custom_model_path
        else:
            if self._model_size not in self.MODEL_SIZES:
                raise ValueError(f"Invalid model size: {model_size}. "
                              f"Choose from {list(self.MODEL_SIZES.keys())}")
            self.model_path = self.MODEL_SIZES[self._model_size]
        
        # Initialize YOLO model
        self.model = YOLO(self.model_path)
        
        # Create box annotator for visualization (optional)
        self.box_annotator = sv.BoxAnnotator()
        
        # Device configuration
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Supervision detector using device: {self.device}")
        
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
        
        # Run YOLO detection
        yolo_results = self.model(image, conf=self.conf_threshold, verbose=False)[0]
        
        # Convert to supervision Detections format
        sv_detections = sv.Detections.from_ultralytics(yolo_results)
        
        # Process results
        detections = []
        
        if len(sv_detections):
            for i in range(len(sv_detections.xyxy)):
                box = sv_detections.xyxy[i]
                class_id = int(sv_detections.class_id[i])
                confidence = float(sv_detections.confidence[i])
                
                # Filter for vehicle classes only
                if class_id in self.vehicle_classes:
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': confidence,
                        'class_id': class_id,
                        'area': float((box[2] - box[0]) * (box[3] - box[1])),
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