"""
YOLOv12 detector implementation
"""
import logging
from typing import List
import os

import numpy as np
from ultralytics import YOLO

from .base_detector import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class Yolov12Detector(BaseDetector):
    """YOLOv12-based detector for object detection"""
    
    def _initialize_model(self) -> None:
        """Initialize YOLOv12 model"""
        try:
            # Get model path from config
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
            
            # Load YOLO model
            self.model = YOLO(model_path)
            
            # Move to device
            self.model.to(self.device)
            
            logger.info(f"Successfully initialized YOLOv12 model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv12 model: {e}")
            raise
    
    def _run_inference(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run YOLOv12 inference on the image
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            List of DetectionResult objects
        """
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            
            # Process results
            for result in results:
                if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                    continue
                
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates [x1, y1, x2, y2]
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    
                    # Get confidence and class
                    confidence = boxes.conf[i].item()
                    class_id = int(boxes.cls[i].item())
                    
                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Create detection result
                    detection = DetectionResult(
                        bbox=[x1, y1, x2, y2],
                        confidence=confidence,
                        class_id=class_id,
                        area=area
                    )
                    
                    detections.append(detection)
            
            logger.debug(f"YOLOv12 detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"YOLOv12 inference failed: {e}")
            return []