"""
Base class for end-to-end zero-shot detection and classification models
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import time
import logging

import numpy as np
import torch
from PIL import Image

from config import END_TO_END_CONFIGS, VEHICLE_CLASSES, MAIN_VEHICLE_SCORING

logger = logging.getLogger(__name__)


class EndToEndDetection:
    """Standardized end-to-end detection result container"""
    
    def __init__(self, bbox: List[float], predicted_class: str, confidence: float, area: float):
        """
        Initialize end-to-end detection result
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            predicted_class: Predicted vehicle class
            confidence: Detection/classification confidence score
            area: Bounding box area in pixels
        """
        self.bbox = self._validate_bbox(bbox)
        self.predicted_class = self._validate_class(predicted_class)
        self.confidence = max(0.0, min(1.0, float(confidence)))
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
    
    def _validate_class(self, predicted_class: str) -> str:
        """Validate that the predicted class is in VEHICLE_CLASSES"""
        if predicted_class not in VEHICLE_CLASSES:
            logger.warning(f"Invalid class '{predicted_class}', defaulting to 'non-vehicle'")
            return "non-vehicle"
        return predicted_class
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'bbox': self.bbox,
            'class': self.predicted_class,
            'score': self.confidence,
            'area': self.area,
            'combined_score': self.score
        }


class EndToEndResult:
    """Container for end-to-end model results"""
    
    def __init__(self, detections: List[EndToEndDetection], main_vehicle: Optional[EndToEndDetection]):
        """
        Initialize end-to-end result
        
        Args:
            detections: List of all detections
            main_vehicle: Main vehicle detection (highest scored)
        """
        self.detections = detections
        self.main_vehicle = main_vehicle
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'detections': [det.to_dict() for det in self.detections],
            'main_vehicle': self.main_vehicle.to_dict() if self.main_vehicle else None
        }


class BaseEndToEndModel(ABC):
    """Abstract base class for end-to-end zero-shot detection and classification models"""
    
    def __init__(self, model_size: str = None, confidence_threshold: float = None,
                 custom_model_path: Optional[str] = None):
        """
        Initialize base end-to-end model
        
        Args:
            model_size: Size of the model to use
            confidence_threshold: Confidence threshold for detections
            custom_model_path: Path to custom model weights
        """
        self.model_name = self.__class__.__name__.replace('Detector', '').lower()
        # Handle special cases for model name mapping
        if self.model_name == 'groundingdino':
            self.model_name = 'dino'
        elif self.model_name == 'yoloworld':
            self.model_name = 'yolo_world'
        
        self.config = END_TO_END_CONFIGS.get(self.model_name, {})
        
        # Set model parameters
        self.model_size = model_size or self.config.get('default_model_size', 'medium')
        self.confidence_threshold = confidence_threshold or self.config.get('confidence_threshold', 0.3)
        self.custom_model_path = custom_model_path
        
        # Set up device
        self.device = self._get_best_device()
        
        # Initialize model (to be implemented by subclasses)
        self.model = None
        self._initialize_model()
        
        logger.info(f"Initialized {self.model_name} end-to-end model with model size: {self.model_size}, "
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
    
    def _validate_image(self, image: np.ndarray) -> Image.Image:
        """
        Validate and convert image to PIL format
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            PIL Image object
        """
        if image is None:
            raise ValueError("Image cannot be None")
        
        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Image array is empty")
            
            if image.ndim not in [2, 3]:
                raise ValueError(f"Image must be 2D or 3D array, got {image.ndim}D")
            
            # Convert to PIL Image
            if image.ndim == 2:  # Grayscale
                pil_image = Image.fromarray(image, mode='L').convert('RGB')
            else:  # RGB
                pil_image = Image.fromarray(image.astype('uint8'), mode='RGB')
            
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Check image dimensions
        if pil_image.size[0] == 0 or pil_image.size[1] == 0:
            raise ValueError(f"Image has invalid dimensions: {pil_image.size}")
        
        return pil_image
    
    def _map_to_vehicle_class(self, label: str) -> str:
        """
        Map model output label to standardized vehicle classes
        
        Args:
            label: Original label from the model
            
        Returns:
            Standardized vehicle class name
        """
        label_lower = label.lower()
        
        # Map to standardized classes
        if any(term in label_lower for term in ["person", "pedestrian", "people", "human"]):
            return "non-vehicle"
        elif any(term in label_lower for term in ["car", "sedan", "coupe", "suv", "automobile"]):
            return "car"
        elif any(term in label_lower for term in ["van", "minivan", "delivery", "panel"]):
            return "van" 
        elif any(term in label_lower for term in ["truck", "pickup", "freight", "cargo"]):
            return "truck"
        elif any(term in label_lower for term in ["bus", "coach", "transit"]):
            return "bus"
        else:
            # Check if it's already a valid class
            if label_lower in [cls.lower() for cls in VEHICLE_CLASSES]:
                return label_lower
            # Default to non-vehicle for unrecognized classes
            return "non-vehicle"
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize the model (to be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _run_inference(self, image: Image.Image) -> List[EndToEndDetection]:
        """
        Run model inference on the image
        
        Args:
            image: Input image as PIL Image
            
        Returns:
            List of EndToEndDetection objects
        """
        pass
    
    def detect_and_classify(self, image: np.ndarray) -> Tuple[Dict, float]:
        """
        Detect and classify objects in the image
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Tuple of (detection/classification results, inference time)
        """
        start_time = time.time()
        
        try:
            # Validate and convert image
            pil_image = self._validate_image(image)
            
            # Run model-specific inference
            detections = self._run_inference(pil_image)
            
            # Find main vehicle
            main_vehicle = self._find_main_vehicle(detections, image.shape)
            
            # Create result
            result = EndToEndResult(detections, main_vehicle)
            
            inference_time = time.time() - start_time
            
            logger.debug(f"Detected {len(detections)} objects, main vehicle: "
                        f"{main_vehicle.predicted_class if main_vehicle else 'None'} in {inference_time:.3f}s")
            
            return result.to_dict(), inference_time
            
        except Exception as e:
            logger.error(f"Error during detection and classification: {e}")
            
            # Return empty result
            empty_result = EndToEndResult([], None)
            return empty_result.to_dict(), time.time() - start_time
    
    def _find_main_vehicle(self, detections: List[EndToEndDetection], 
                          image_shape: Tuple[int, int, int]) -> Optional[EndToEndDetection]:
        """
        Find the main vehicle in the image based on centrality and size
        
        Args:
            detections: List of EndToEndDetection objects
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Main vehicle detection, or None
        """
        if not detections:
            logger.debug("No detections found, returning None for main vehicle")
            return None
        
        # Filter out non-vehicles first
        vehicle_detections = [det for det in detections if det.predicted_class != 'non-vehicle']
        
        if not vehicle_detections:
            logger.debug("No vehicle detections found")
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
        
        # Calculate score for each vehicle detection
        scored_detections = []
        
        for det in vehicle_detections:
            # Skip detections that are too small
            if det.area < min_area:
                continue
            
            # Calculate bbox center
            bbox_center = ((det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2)
            
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
            size_score = det.area / img_area
            
            # Combined score
            combined_score = centrality_weight * centrality + size_weight * size_score
            
            # Update detection with score
            det.score = combined_score
            scored_detections.append(det)
        
        if not scored_detections:
            logger.debug("No valid vehicle detections after filtering")
            return None
        
        # Sort by combined score (descending) and return the best
        scored_detections.sort(key=lambda x: x.score, reverse=True)
        main_vehicle = scored_detections[0]
        
        logger.debug(f"Found main vehicle: {main_vehicle.predicted_class} with score: {main_vehicle.score:.3f}")
        
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