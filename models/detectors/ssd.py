"""
SSD detector implementation
"""
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torchvision
import cv2

from config import DETECTOR_CONFIGS


class SSDDetector:
    """SSD-based detector for object detection"""
    
    MODEL_SIZES = {
        "small": "ssd_mobilenet_v2",
        "medium": "ssd_resnet50_fpn",
    }
    
    def __init__(self, model_size: str = "small", conf_threshold: float = None, custom_model_path: Optional[str] = None):
        """
        Initialize SSD detector
        
        Args:
            model_size: Size of the model (small=mobilenet, medium=resnet)
            conf_threshold: Confidence threshold for detections
            custom_model_path: Path to custom model weights (overrides model_size if provided)
        """
        # Use config values if not provided
        config = DETECTOR_CONFIGS.get('ssd', {
            'model_size': 'small',
            'conf_threshold': 0.3,
            'vehicle_classes': [2, 3, 5, 7]  # COCO class IDs for vehicles
        })
        
        self._model_size = model_size
        self.conf_threshold = conf_threshold or config.get('conf_threshold', 0.3)
        self.vehicle_classes = config.get('vehicle_classes', [2, 3, 5, 7])
        
        # Load model
        if custom_model_path and os.path.exists(custom_model_path):
            # Load custom model if provided
            self.model = torch.load(custom_model_path)
        else:
            if self._model_size not in self.MODEL_SIZES:
                raise ValueError(f"Invalid model size: {model_size}. "
                              f"Choose from {list(self.MODEL_SIZES.keys())}")
            
            # Load pretrained model from torchvision
            model_name = self.MODEL_SIZES[self._model_size]
            print(f"Loading SSD model: {model_name}")
            
            if model_name == "ssd_mobilenet_v2":
                self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
            else:  # resnet
                self.model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Device configuration
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"SSD detector using device: {self.device}")
        
    def detect(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Detect objects in the image
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Tuple of (list of detections, inference time)
        """
        # Start timing
        start_time = time.time()
        
        # Convert to tensor and normalize
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Extract results from first image in batch
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Process results
        detections = []
        
        for i in range(len(boxes)):
            # Filter by confidence
            if scores[i] < self.conf_threshold:
                continue
                
            # Get class ID (SSD returns COCO class IDs)
            class_id = int(labels[i]) - 1  # SSD class IDs are 1-indexed, COCO is 0-indexed
            
            # Only keep vehicle classes
            if class_id in self.vehicle_classes:
                x1, y1, x2, y2 = boxes[i]
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(scores[i]),
                    'class_id': class_id,
                    'area': float((x2 - x1) * (y2 - y1)),
                })
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        return detections, inference_time
    
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
            det['score'] = 0.7 * centrality + 0.3 * size_score
        
        # Sort by combined score (descending)
        detections.sort(key=lambda x: x['score'], reverse=True)
        
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