"""
SSD detector implementation
"""
import logging
from typing import List
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from .base_detector import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class SsdDetector(BaseDetector):
    """SSD-based detector for object detection"""
    
    def _initialize_model(self) -> None:
        """Initialize SSD model"""
        try:
            # Get model configuration
            model_sizes = self.config.get('model_sizes', {})
            
            if self.custom_model_path and os.path.exists(self.custom_model_path):
                # Load custom model
                self.model = torch.load(self.custom_model_path, map_location=self.device)
                logger.info(f"Loaded custom SSD model from: {self.custom_model_path}")
            elif self.model_size in model_sizes:
                model_name = model_sizes[self.model_size]
                
                # Load pretrained SSD model based on configuration
                if model_name == "ssd_mobilenet_v3":
                    self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
                        weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
                    )
                elif model_name == "ssd_vgg16":
                    self.model = torchvision.models.detection.ssd300_vgg16(
                        weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
                    )
                else:
                    raise ValueError(f"Unsupported SSD model: {model_name}")
                
                logger.info(f"Loaded pretrained SSD model: {model_name}")
            else:
                raise ValueError(f"Invalid model size '{self.model_size}'. "
                               f"Available sizes: {list(model_sizes.keys())}")
            
            # Set model to evaluation mode and move to device
            self.model.eval()
            self.model.to(self.device)
            
            # Define image preprocessing transforms
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            logger.info(f"Successfully initialized SSD model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SSD model: {e}")
            raise
    
    def _run_inference(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run SSD inference on the image
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            List of DetectionResult objects
        """
        try:
            # Convert image to tensor and add batch dimension
            if image.dtype != np.float32:
                # Normalize to [0, 1] if not already
                image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
            else:
                image_tensor = torch.from_numpy(image)
            
            # Ensure tensor is in CHW format
            if image_tensor.dim() == 3 and image_tensor.shape[2] == 3:
                image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
            
            # Add batch dimension and move to device
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            detections = []
            
            # Process predictions
            if predictions and len(predictions) > 0:
                pred = predictions[0]
                
                boxes = pred.get('boxes', torch.empty(0)).cpu().numpy()
                scores = pred.get('scores', torch.empty(0)).cpu().numpy()
                labels = pred.get('labels', torch.empty(0)).cpu().numpy()
                
                for i in range(len(boxes)):
                    confidence = float(scores[i])
                    
                    # Filter by confidence threshold
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Get box coordinates [x1, y1, x2, y2]
                    x1, y1, x2, y2 = boxes[i]
                    
                    # Get class ID
                    class_id = int(labels[i])
                    
                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Create detection result
                    detection = DetectionResult(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        confidence=confidence,
                        class_id=class_id,
                        area=float(area)
                    )
                    
                    detections.append(detection)
            
            logger.debug(f"SSD detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"SSD inference failed: {e}")
            return []