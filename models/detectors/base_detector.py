"""
Base detector class for object detection models
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import time
import logging

import numpy as np
import torch

from config import DETECTOR_CONFIGS, MAIN_VEHICLE_SCORING

logger = logging.getLogger(__name__)


class DetectionResult:
    """Standardized detection result container"""
    
    def __init__(self, bbox: List[float], confidence: float, class_id: int, area: float):
        """
        Initialize detection result
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            confidence: Detection confidence score
            class_id: Class ID from the model
            area: Bounding box area in pixels
        """
        self.bbox = self._validate_bbox(bbox)
        self.confidence = max(0.0, min(1.0, float(confidence)))
        self.class_id = int(class_id)
        self.area = max(0.0, float(area))
        self.score = 0.0  # Will be computed by find_main_vehicle
        
    def _validate_bbox(self, bbox: List[float]) -> List[float]:
        """Validate and fix bounding box coordinates"""
        if len(bbox) != 4:
            raise ValueError(f"Bounding box must have 4 coordinates, got {len(bbox)}")
        
        x1, y1, x2, y2 = map(float, bbox)
        
        # Ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        return [x1, y1, x2, y2]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'area': self.area,
            'score': self.score
        }


class BaseDetector(ABC):
    """Abstract base class for object detectors"""
    
    def __init__(self, model_size: str = None, confidence_threshold: float = None, 
                 custom_model_path: Optional[str] = None):
        """
        Initialize base detector
        
        Args:
            model_size: Size of the model to use
            confidence_threshold: Confidence threshold for detections
            custom_model_path: Path to custom model weights
        """
        self.detector_name = self.__class__.__name__.replace('Detector', '').lower()
        self.config = DETECTOR_CONFIGS.get(self.detector_name, {})
        
        # Set model parameters
        self.model_size = model_size or self.config.get('default_model_size', 'medium')
        self.confidence_threshold = confidence_threshold or self.config.get('confidence_threshold', 0.3)
        self.custom_model_path = custom_model_path
        
        # Get vehicle class IDs for filtering
        self.vehicle_class_ids = set(self.config.get('vehicle_class_ids', []))
        
        # Set up device
        self.device = self._get_best_device()
        
        # Initialize model (to be implemented by subclasses)
        self.model = None
        self._initialize_model()
        
        logger.info(f"Initialized {self.detector_name} detector with model size: {self.model_size}, "
                   f"confidence threshold: {self.confidence_threshold}, device: {self.device}")
    
    def _get_best_device(self) -> str:
        """Get the best available device based on configuration preferences"""
        device_preferences = self.config.get('device_preference', ['cuda', 'mps', 'cpu'])
        
        for device in device_preferences:
            if device == 'cuda' and torch.cuda.is_available():
                return 'cuda'
            elif device == 'mps' and torch.backends.mps.is_available():
                return 'mps'
            elif device == 'cpu':
                return 'cpu'
        
        return 'cpu'  # Fallback
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize the detection model (to be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _run_inference(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run model inference on the image
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            List of DetectionResult objects
        """
        pass
    
    def detect(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Detect objects in the image
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Tuple of (list of detection dictionaries, inference time)
        """
        if image is None or image.size == 0:
            logger.warning("Received invalid image for detection")
            return [], 0.0
        
        start_time = time.time()
        
        try:
            # Run model-specific inference
            detections = self._run_inference(image)
            
            # Filter for vehicle classes only
            vehicle_detections = [
                det for det in detections 
                if det.class_id in self.vehicle_class_ids
            ]
            
            # Convert to dictionary format
            detection_dicts = [det.to_dict() for det in vehicle_detections]
            
            inference_time = time.time() - start_time
            
            logger.debug(f"Detected {len(detection_dicts)} vehicles in {inference_time:.3f}s")
            
            return detection_dicts, inference_time
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return [], time.time() - start_time
    
    def find_main_vehicle(self, detections: List[Dict], image_shape: Tuple[int, int, int]) -> Optional[Dict]:
        """
        Find the main vehicle in the image based on centrality and size
        
        Args:
            detections: List of detection dictionaries
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Dictionary containing the main vehicle detection, or None
        """
        if not detections:
            logger.debug("No detections found, returning None for main vehicle")
            return None
        
        img_height, img_width = image_shape[:2]
        img_center = (img_width / 2, img_height / 2)
        img_area = img_width * img_height
        img_diagonal = np.sqrt(img_width**2 + img_height**2)
        
        # Get scoring weights from config
        centrality_weight = MAIN_VEHICLE_SCORING['centrality_weight']
        size_weight = MAIN_VEHICLE_SCORING['size_weight']
        min_area = MAIN_VEHICLE_SCORING['min_area_threshold']
        max_distance = MAIN_VEHICLE_SCORING['max_distance_threshold']
        
        # Calculate score for each detection
        scored_detections = []
        
        for det in detections:
            bbox = det['bbox']
            area = det['area']
            
            # Skip detections that are too small
            if area < min_area:
                continue
            
            # Calculate bbox center
            bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # Calculate distance from image center
            distance = np.sqrt((bbox_center[0] - img_center[0])**2 + 
                             (bbox_center[1] - img_center[1])**2)
            
            # Normalize distance by image diagonal
            normalized_distance = distance / img_diagonal
            
            # Skip detections that are too far from center
            if normalized_distance > max_distance:
                continue
            
            # Calculate centrality score (1 - normalized_distance)
            centrality = 1 - normalized_distance
            
            # Calculate size score (normalized by image area)
            size_score = area / img_area
            
            # Combined score
            combined_score = centrality_weight * centrality + size_weight * size_score
            
            # Update detection with score
            det['score'] = combined_score
            scored_detections.append(det)
        
        if not scored_detections:
            logger.debug("No valid detections after filtering")
            return None
        
        # Sort by combined score (descending) and return the best
        scored_detections.sort(key=lambda x: x['score'], reverse=True)
        main_vehicle = scored_detections[0]
        
        logger.debug(f"Found main vehicle with score: {main_vehicle['score']:.3f}")
        
        return main_vehicle
    
    def crop_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Crop image to the specified bounding box
        
        Args:
            image: Input image (RGB format, numpy array)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Cropped image
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided for cropping")
        
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(x1, min(x2, w))
        y2 = max(y1, min(y2, h))
        
        # Check if the crop area is valid
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop coordinates: ({x1}, {y1}, {x2}, {y2})")
        
        # Crop and return
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size == 0:
            raise ValueError("Cropped image is empty")
        
        return cropped