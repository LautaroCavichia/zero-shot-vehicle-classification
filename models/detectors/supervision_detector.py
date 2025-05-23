"""
Supervision detector implementation
"""
import logging
from typing import List
import os

import numpy as np
import supervision as sv
from ultralytics import YOLO

from .base_detector import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class SupervisionDetector(BaseDetector):
    """Supervision-based detector wrapper for YOLO models"""
    
    def _initialize_model(self) -> None:
        """Initialize Supervision detector with YOLO backend"""
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
            
            # Initialize YOLO model for supervision
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Create supervision box annotator (for future use)
            self.box_annotator = sv.BoxAnnotator()
            
            logger.info(f"Successfully initialized Supervision detector on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supervision detector: {e}")
            raise
    
    def _run_inference(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run Supervision inference on the image
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            List of DetectionResult objects
        """
        try:
            # Run YOLO detection
            yolo_results = self.model(image, conf=self.confidence_threshold, verbose=False)[0]
            
            # Convert to supervision Detections format
            sv_detections = sv.Detections.from_ultralytics(yolo_results)
            
            detections = []
            
            # Process supervision detections
            if len(sv_detections):
                for i in range(len(sv_detections.xyxy)):
                    # Get box coordinates [x1, y1, x2, y2]
                    box = sv_detections.xyxy[i]
                    x1, y1, x2, y2 = box.tolist()
                    
                    # Get class ID and confidence
                    class_id = int(sv_detections.class_id[i])
                    confidence = float(sv_detections.confidence[i])
                    
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
            
            logger.debug(f"Supervision detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Supervision inference failed: {e}")
            return []