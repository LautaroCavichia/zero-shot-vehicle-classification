"""
YOLO-World model implementation for zero-shot object detection and classification
"""
import logging
from typing import List
import os

import numpy as np
from PIL import Image
from ultralytics import YOLO

from .base_end_to_end import BaseEndToEndModel, EndToEndDetection
from config import VEHICLE_CLASSES

logger = logging.getLogger(__name__)


class YoloWorldDetector(BaseEndToEndModel):
    """YOLO-World detector for end-to-end zero-shot object detection and classification"""
    
    def _initialize_model(self) -> None:
        """Initialize YOLO-World model"""
        try:
            # Get model configuration
            model_sizes = self.config.get('model_sizes', {})
            
            if self.custom_model_path and os.path.exists(self.custom_model_path):
                model_path = self.custom_model_path
                logger.info(f"Using custom model path: {model_path}")
            elif self.model_size in model_sizes:
                model_path = model_sizes[self.model_size]
                logger.info(f"Using model: {model_path}")
            else:
                raise ValueError(f"Invalid model size '{self.model_size}'. "
                               f"Available sizes: {list(model_sizes.keys())}")
            
            # Initialize YOLO-World model
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Set the classes for zero-shot detection
            self.model.set_classes(VEHICLE_CLASSES)
            
            logger.info(f"Successfully initialized YOLO-World model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO-World model: {e}")
            raise
    
    def _run_inference(self, image: Image.Image) -> List[EndToEndDetection]:
        """
        Run YOLO-World inference on the image
        
        Args:
            image: Input image as PIL Image
            
        Returns:
            List of EndToEndDetection objects
        """
        try:
            # Convert PIL image to numpy array for YOLO
            image_array = np.array(image)
            
            # Run YOLO-World detection
            results = self.model.predict(
                source=image_array,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            detections = []
            
            # Process detection results
            for result in results:
                if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                    continue
                
                for i in range(len(result.boxes)):
                    # Get box coordinates [x1, y1, x2, y2]
                    x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                    
                    # Get class information
                    cls_idx = int(result.boxes.cls[i].item())
                    confidence = result.boxes.conf[i].item()
                    
                    # Get the class name
                    if hasattr(result, 'names') and cls_idx in result.names:
                        original_class = result.names[cls_idx]
                    else:
                        original_class = f"class_{cls_idx}"
                    
                    # Map to standardized vehicle classes
                    mapped_class = self._map_to_vehicle_class(original_class)
                    
                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Create detection
                    detection = EndToEndDetection(
                        bbox=[x1, y1, x2, y2],
                        predicted_class=mapped_class,
                        confidence=confidence,
                        area=area
                    )
                    
                    detections.append(detection)
            
            logger.debug(f"YOLO-World detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"YOLO-World inference failed: {e}")
            # Try fallback inference without setting classes
            try:
                logger.warning("Attempting fallback inference without class setting")
                results = self.model.predict(
                    source=np.array(image),
                    conf=self.confidence_threshold,
                    verbose=False
                )
                
                detections = []
                for result in results:
                    if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                        continue
                    
                    for i in range(len(result.boxes)):
                        x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                        cls_idx = int(result.boxes.cls[i].item())
                        confidence = result.boxes.conf[i].item()
                        
                        # Default mapping for fallback
                        mapped_class = "non-vehicle"  # Conservative fallback
                        area = (x2 - x1) * (y2 - y1)
                        
                        detection = EndToEndDetection(
                            bbox=[x1, y1, x2, y2],
                            predicted_class=mapped_class,
                            confidence=confidence,
                            area=area
                        )
                        
                        detections.append(detection)
                
                logger.debug(f"Fallback inference detected {len(detections)} objects")
                return detections
                
            except Exception as fallback_error:
                logger.error(f"Fallback inference also failed: {fallback_error}")
                return []