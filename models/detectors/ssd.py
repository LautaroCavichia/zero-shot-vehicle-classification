"""
SSD detector implementation with correct PyTorch COCO class IDs
"""
import logging
from typing import List
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from .base_detector import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class SsdDetector(BaseDetector):
    """SSD-based detector with correct PyTorch COCO class IDs"""
    
    def _initialize_model(self) -> None:
        """Initialize SSD model with proper weights and class mapping"""
        try:
            # Get model configuration
            model_sizes = self.config.get('model_sizes', {})
            
            if self.custom_model_path and os.path.exists(self.custom_model_path):
                # Load custom model
                self.model = torch.load(self.custom_model_path, map_location=self.device)
                logger.info(f"Loaded custom SSD model from: {self.custom_model_path}")
            elif self.model_size in model_sizes:
                model_name = model_sizes[self.model_size]
                
                # Load pretrained SSD model with proper weights
                if model_name == "ssd_mobilenet_v3":
                    from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
                    
                    self.model = ssdlite320_mobilenet_v3_large(
                        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
                    )
                    
                elif model_name == "ssd_vgg16":
                    from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
                    
                    self.model = ssd300_vgg16(
                        weights=SSD300_VGG16_Weights.COCO_V1
                    )
                    
                else:
                    raise ValueError(f"Unsupported SSD model: {model_name}")
                
                logger.info(f"Loaded pretrained SSD model: {model_name} with COCO_V1 weights")
            else:
                raise ValueError(f"Invalid model size '{self.model_size}'. "
                               f"Available sizes: {list(model_sizes.keys())}")
            
            # Set model to evaluation mode and move to device
            self.model.eval()
            self.model.to(self.device)
            
            # Define proper preprocessing transforms
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Converts PIL to tensor and scales to [0,1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225]   # ImageNet stds
                )
            ])
            
            # Log the correct COCO class mapping for debugging
            logger.info("SSD using PyTorch COCO class IDs:")
            logger.info("  1: person, 2: bicycle, 3: car, 4: motorcycle")
            logger.info("  6: bus, 7: train, 8: truck")
            
            logger.info(f"Successfully initialized SSD model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SSD model: {e}")
            raise
    
    def _run_inference(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run SSD inference on the image with proper preprocessing
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            List of DetectionResult objects
        """
        try:
            # Convert numpy array to PIL Image for proper preprocessing
            if isinstance(image, np.ndarray):
                # Ensure image is in uint8 format
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                
                # Convert to PIL
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Apply transforms: PIL -> Tensor -> Normalized
            image_tensor = self.transform(pil_image)
            
            # Add batch dimension and move to device
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_batch)
            
            detections = []
            
            # Process predictions
            if predictions and len(predictions) > 0:
                pred = predictions[0]
                
                boxes = pred.get('boxes', torch.empty(0)).cpu().numpy()
                scores = pred.get('scores', torch.empty(0)).cpu().numpy()
                labels = pred.get('labels', torch.empty(0)).cpu().numpy()
                
                # Debug: Log unique class IDs found
                unique_classes = set(labels.astype(int)) if len(labels) > 0 else set()
                if unique_classes:
                    logger.debug(f"SSD detected classes: {unique_classes}")
                
                for i in range(len(boxes)):
                    confidence = float(scores[i])
                    
                    # Filter by confidence threshold
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Get box coordinates [x1, y1, x2, y2]
                    x1, y1, x2, y2 = boxes[i]
                    
                    # Get class ID (PyTorch SSD COCO format)
                    class_id = int(labels[i])
                    
                    # Debug: Log detection details for vehicle classes
                    if class_id in self.vehicle_class_ids:
                        class_name_map = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 
                                        6: "bus", 7: "train", 8: "truck"}
                        class_name = class_name_map.get(class_id, f"class_{class_id}")
                        logger.debug(f"SSD detected {class_name} (ID:{class_id}) with conf:{confidence:.3f}")
                    
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