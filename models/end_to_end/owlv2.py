"""
OWLv2 model implementation for zero-shot object detection and classification
"""
import time
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
import cv2

from config import END_TO_END_CONFIGS, VEHICLE_CLASSES


class OWLv2Detector:
    """OWLv2 detector for zero-shot object detection and classification"""
    
    MODEL_SIZES = {
        "base": "google/owlv2-base-patch16",
        "large": "google/owlv2-large-patch14",
        "base-ensemble": "google/owlv2-base-patch16-ensemble"
    }
    
    def __init__(self, model_size: str = "base", confidence_threshold: float = None, custom_model_path: Optional[str] = None):
        """
        Initialize OWLv2 detector
        
        Args:
            model_size: Size of the model (base, large, base-ensemble)
            confidence_threshold: Confidence threshold for detections
            custom_model_path: Path to custom model weights (overrides model_size if provided)
        """
        # Use config values if not provided
        config = END_TO_END_CONFIGS.get('owlv2', {'confidence_threshold': 0.25})
        self._model_size = model_size.lower()
        self.confidence_threshold = confidence_threshold or config.get('confidence_threshold', 0.25)
        
        # Set up model path
        if custom_model_path:
            self.model_path = custom_model_path
        else:
            if self._model_size not in self.MODEL_SIZES:
                raise ValueError(f"Invalid model size: {self._model_size}. "
                              f"Choose from {list(self.MODEL_SIZES.keys())}")
            self.model_path = self.MODEL_SIZES[self._model_size]
        
        # Device configuration
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"OWLv2 detector using device: {self.device}")
        self._init_model()
        
    def _init_model(self):
        """Initialize OWLv2 model"""
        try:
            # Import using exactly the same imports as in the example
            from transformers import pipeline
            
            self.detector = pipeline(
                model=self.model_path, 
                task="zero-shot-object-detection", 
                device=self.device
            )
            
            # Store classes for detection
            self.vehicle_classes = VEHICLE_CLASSES
            print(f"OWLv2 model initialized with classes: {self.vehicle_classes}")
            
        except Exception as e:
            print(f"Error initializing OWLv2 model: {e}")
            raise
    
    def detect_and_classify(self, image: np.ndarray) -> Tuple[Dict, float]:
        """
        Detect and classify objects in the image using OWLv2
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Tuple of (detection/classification results, inference time)
        """
        # Start timing
        start_time = time.time()
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image).convert("RGB")
        
        # Run OWLv2 detection
        try:
            predictions = self.detector(
                pil_image,
                candidate_labels=self.vehicle_classes,
                threshold=self.confidence_threshold
            )
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return empty results on error
            return {'detections': [], 'main_vehicle': None}, time.time() - start_time
        
        # Process results into the same format as YOLO-World
        processed_detections = []
        
        # Process detection results
        for pred in predictions:
            # Get box coordinates (convert to x1, y1, x2, y2 format)
            box = pred['box']
            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            
            # Get class information and confidence score
            class_name = pred['label']
            score = pred['score']
            
            # Standardize class name
            standard_class = self._map_to_vehicle_class(class_name)
            
            # Calculate area
            area = (x2 - x1) * (y2 - y1)
            
            processed_detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'class': standard_class,
                'original_class': class_name,
                'score': float(score),
                'area': float(area),
            })
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Find main vehicle
        main_vehicle = self.find_main_vehicle(processed_detections, image.shape)
        
        return {
            'detections': processed_detections,
            'main_vehicle': main_vehicle,
        }, inference_time
    
    def _map_to_vehicle_class(self, label: str) -> str:
        """Map the model's output label to standard vehicle classes"""
        # For OWLv2, the output label should already be one of our vehicle classes
        # since we provide the candidate_labels
        
        # But let's still do some standardization just in case
        label_lower = label.lower()
        
        if "person" in label_lower or "non-vehicle" in label_lower:
            return "non-vehicle"
        elif any(x in label_lower for x in ["car", "sedan", "coupe", "suv"]):
            return "car"
        elif "van" in label_lower:
            return "van"
        elif "truck" in label_lower:
            return "truck"
        elif "bus" in label_lower:
            return "bus"
        elif any(x in label_lower for x in ["ambulance", "police", "emergency", "fire"]):
            return "emergency"
        
        # The label should already be in our vehicle_classes list, so return as is
        if label in self.vehicle_classes:
            return label
            
        # Fallback
        return "non-vehicle"
    
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
        
        # Define scoring parameters
        center_weight = 2.0    # Weight for centrality
        size_weight = 1.1      # Weight for size
        emergency_bonus = 1.5  # Bonus for emergency vehicles
        
        # Calculate score for each detection
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
            
            # Apply emergency vehicle bonus
            emergency_factor = emergency_bonus if det['class'] == 'emergency' else 1.0
            
            # Combined score (balance of centrality and size with emergency bonus)
            det['combined_score'] = (center_weight * centrality + size_weight * size_score) * emergency_factor
        
        # Sort by combined score (descending)
        detections.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Filter out non-vehicles
        vehicles = [det for det in detections if det['class'] != 'non-vehicle']
        
        # Return the highest scored vehicle, or highest scored detection if no vehicles
        return vehicles[0] if vehicles else (detections[0] if detections else None)
    
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