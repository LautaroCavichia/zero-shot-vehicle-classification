"""
YOLO-World model implementation for zero-shot object detection and classification
"""
import time
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import cv2
from PIL import Image
from ultralytics import YOLO

from config import END_TO_END_CONFIGS, VEHICLE_CLASSES


class YOLOWorldDetector:
    """YOLO-World detector for end-to-end zero-shot object detection and classification"""
    
    MODEL_SIZES = {
        "small": "yolov8s-worldv2.pt",
        "medium": "yolov8m-worldv2.pt",
        "large": "yolov8l-worldv2.pt"
    }
    
    def __init__(self, model_size: str = "small", confidence_threshold: float = None, custom_model_path: Optional[str] = None):
        """
        Initialize YOLO-World detector
        
        Args:
            model_size: Size of the model (small, medium, large)
            confidence_threshold: Confidence threshold for detections
            custom_model_path: Path to custom model weights (overrides model_size if provided)
        """
        # Use config values if not provided
        config = END_TO_END_CONFIGS['yolo_world']
        self._model_size = model_size.lower()
        self.confidence_threshold = confidence_threshold or config['confidence_threshold']
        
        # Set up model path
        if custom_model_path and os.path.exists(custom_model_path):
            self.model_path = custom_model_path
        else:
            if self._model_size not in self.MODEL_SIZES:
                raise ValueError(f"Invalid model size: {self._model_size}. "
                              f"Choose from {list(self.MODEL_SIZES.keys())}")
            self.model_path = self.MODEL_SIZES[self._model_size]
        

        # Device configuration
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"YOLO-World detector using device: {self.device}")
        self._init_model()
        
    def _init_model(self):
        """Initialize YOLO-World model"""
        # Initialize YOLO-World model using Ultralytics implementation
        self.model = YOLO(self.model_path)
        
        # Set device
        self.model.to(self.device)
        classes = VEHICLE_CLASSES
        self.model.set_classes(classes)
    
    
    def detect_and_classify(self, image: np.ndarray) -> Tuple[Dict, float]:
        """
        Detect and classify objects in the image using YOLO-World
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Tuple of (detection/classification results, inference time)
        """
        # Start timing
        start_time = time.time()
        
        # Run YOLO-World detection
        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                verbose=False
            )
        except Exception as e:
            print(f"Prediction error: {e}")
            print("Falling back to basic detection...")
            # Try to reset classes and run again
            try:
                self.model.set_classes(VEHICLE_CLASSES)
                results = self.model.predict(
                    source=image,
                    conf=self.confidence_threshold,
                    verbose=False
                )
            except:
                # Last resort - just run without setting classes
                results = self.model.predict(
                    source=image,
                    conf=self.confidence_threshold,
                    verbose=False
                )
        
        # Process results
        processed_detections = []
        
        # Process detection results
        for result in results:
            if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                continue
                
            for i in range(len(result.boxes)):
                # Get box coordinates (convert to x1, y1, x2, y2 format)
                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                
                # Get class information
                cls_idx = int(result.boxes.cls[i].item())
                
                # Get the class name
                if hasattr(result, 'names') and cls_idx in result.names:
                    original_class = result.names[cls_idx]
                else:
                    original_class = f"class_{cls_idx}"
                
                # Map to our vehicle classes
                if "person" in original_class.lower() or "non-vehicle" in original_class.lower():
                    class_name = "non-vehicle"
                elif any(x in original_class.lower() for x in ["car", "sedan", "coupe", "suv"]):
                    class_name = "car"
                elif "van" in original_class.lower():
                    class_name = "van"
                elif "truck" in original_class.lower():
                    class_name = "truck"
                elif "bus" in original_class.lower():
                    class_name = "bus"
                elif any(x in original_class.lower() for x in ["ambulance", "police", "emergency", "fire"]):
                    class_name = "emergency"
                else:
                    # Try to find the best match
                    for cls in VEHICLE_CLASSES:
                        if cls.lower() in original_class.lower():
                            class_name = cls
                            break
                    else:
                        class_name = "non-vehicle"
                
                # Get confidence score
                conf = result.boxes.conf[i].item()
                
                processed_detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class': class_name,
                    'original_class': original_class,
                    'score': float(conf),
                    'area': float((x2 - x1) * (y2 - y1)),
                })
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Find main vehicle
        main_vehicle = self.find_main_vehicle(processed_detections, image.shape)
        
        return {
            'detections': processed_detections,
            'main_vehicle': main_vehicle,
        }, inference_time
    
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
            det['combined_score'] = 0.7 * centrality + 0.3 * size_score
        
        # Sort by combined score (descending)
        detections.sort(key=lambda x: x['combined_score'], reverse=True)
        
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