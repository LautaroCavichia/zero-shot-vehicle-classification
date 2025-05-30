"""
Conservative YOLO-World fix - minimal changes from working version
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
    """YOLO-World detector with conservative improvements"""
    
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
            
            # Use simpler, more focused class prompts
            self.yolo_world_classes = self._create_simple_focused_prompts()
            
            # Set the classes for zero-shot detection
            self.model.set_classes(self.yolo_world_classes)
            
            logger.info(f"Successfully initialized YOLO-World model on {self.device}")
            logger.info(f"Using class prompts: {self.yolo_world_classes}")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO-World model: {e}")
            raise
    
    def _create_simple_focused_prompts(self) -> List[str]:
        """
        Create simple, focused prompts - avoid over-engineering
        Based on what was working better in your 57% version
        """
        # Keep it simple - one primary prompt per class, avoid confusion
        class_prompts = []
        
        if "city_car" in VEHICLE_CLASSES:
            class_prompts.extend([
                "car",           # Simple and direct
                "sedan",         # Specific type
                "small car"      # Size indicator
            ])
        
        if "large_suv" in VEHICLE_CLASSES:
            class_prompts.extend([
                "SUV",           # Clear and simple
                "pickup truck",  # Alternative type
                "large vehicle"  # Size-based
            ])
        
        if "van" in VEHICLE_CLASSES:
            class_prompts.extend([
                "van",           # Direct
                "delivery van",  # Specific use
                "minivan"       # Type variant
            ])
        
        if "truck" in VEHICLE_CLASSES:
            class_prompts.extend([
                "truck",         # Simple
                "cargo truck",   # Specific
                "large truck"    # Size-based
            ])
        
        if "bus" in VEHICLE_CLASSES:
            class_prompts.extend([
                "bus",           # Working perfectly, keep simple
                "public bus"     # Context
            ])
        
        if "motorcycle" in VEHICLE_CLASSES:
            class_prompts.extend([
                "motorcycle",    # Working well, keep simple
                "motorbike"      # Alternative term
            ])
        
        if "non-vehicle" in VEHICLE_CLASSES:
            class_prompts.extend([
                "person",        # Direct
                "pedestrian"     # Context-specific
            ])
        
        return class_prompts
    
    def _map_to_vehicle_class(self, label: str) -> str:
        """
        Simplified mapping - avoid over-complex logic that caused collapse
        """
        label_lower = label.lower().strip()
        
        # Simple direct mapping first
        if label_lower in ["car", "sedan", "small car"]:
            return "city_car"
        elif label_lower in ["suv", "pickup truck", "large vehicle"]:
            return "large_suv"
        elif label_lower in ["van", "delivery van", "minivan"]:
            return "van"
        elif label_lower in ["truck", "cargo truck", "large truck"]:
            return "truck"
        elif label_lower in ["bus", "public bus"]:
            return "bus"
        elif label_lower in ["motorcycle", "motorbike"]:
            return "motorcycle"
        elif label_lower in ["person", "pedestrian"]:
            return "non-vehicle"
        
        # Fallback with simple keyword matching
        elif "car" in label_lower and "large" not in label_lower:
            return "city_car"
        elif any(word in label_lower for word in ["suv", "pickup"]):
            return "large_suv"
        elif "van" in label_lower:
            return "van"
        elif "truck" in label_lower:
            return "truck"
        elif "bus" in label_lower:
            return "bus"
        elif any(word in label_lower for word in ["motorcycle", "bike"]):
            return "motorcycle"
        elif any(word in label_lower for word in ["person", "human", "pedestrian"]):
            return "non-vehicle"
        
        # Conservative default - most common class in your dataset
        logger.debug(f"Unknown class '{label}', defaulting to city_car")
        return "city_car"
    
    def _run_inference(self, image: Image.Image) -> List[EndToEndDetection]:
        """
        Simplified inference - remove aggressive filtering that caused problems
        """
        try:
            # Convert PIL image to numpy array for YOLO
            image_array = np.array(image)
            
            # Use moderate confidence threshold - not too aggressive
            results = self.model.predict(
                source=image_array,
                conf=0.15,  # Moderate threshold, not too low
                iou=0.7,    # Higher IoU to reduce duplicates
                max_det=20, # Reasonable limit
                verbose=False,
                agnostic_nms=True  # Use agnostic NMS to reduce confusion
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
                    elif cls_idx < len(self.yolo_world_classes):
                        original_class = self.yolo_world_classes[cls_idx]
                    else:
                        original_class = f"class_{cls_idx}"
                    
                    # Map to standardized vehicle classes
                    mapped_class = self._map_to_vehicle_class(original_class)
                    
                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Simple area filter - minimum threshold only
                    if area < 130:  
                        continue
                    
                    # Create detection
                    detection = EndToEndDetection(
                        bbox=[x1, y1, x2, y2],
                        predicted_class=mapped_class,
                        confidence=confidence,
                        area=area
                    )
                    
                    detections.append(detection)
                    
                    logger.debug(f"YOLO-World: {original_class} -> {mapped_class} (conf: {confidence:.3f})")
            
            logger.debug(f"YOLO-World detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"YOLO-World inference failed: {e}")
            return []