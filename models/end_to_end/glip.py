"""
GLIP (Grounded Language-Image Pretraining) model implementation
"""
import time
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import cv2
from PIL import Image

# Add GLIP to path
sys.path.append("GLIP")

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

from config import END_TO_END_CONFIGS, VEHICLE_CLASSES


class GLIPDetector:
    """GLIP detector for end-to-end zero-shot object detection and classification"""
    
    def __init__(self, config_file: str = None, weight_file: str = None, confidence_threshold: float = None):
        """
        Initialize GLIP detector
        
        Args:
            config_file: Path to the GLIP config file
            weight_file: Path to the GLIP model weights
            confidence_threshold: Confidence threshold for detections
        """
        # Use config values if not provided
        config = END_TO_END_CONFIGS['glip']
        self.config_file = config_file or config['config_file']
        self.weight_file = weight_file or config['weight_file']
        self.confidence_threshold = confidence_threshold or config['confidence_threshold']
        
        # Load model
        self._init_model()
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"GLIP detector using device: {self.device}")
        
    def _init_model(self):
        """Initialize GLIP model"""
        # Update the config options with the config file
        cfg.local_rank = 0
        cfg.num_gpus = 1
        cfg.merge_from_file(self.config_file)
        cfg.merge_from_list(["MODEL.WEIGHT", self.weight_file])
        cfg.merge_from_list(["MODEL.DEVICE", "cuda" if torch.cuda.is_available() else "cpu"])
        
        # Initialize GLIP demo
        self.model = GLIPDemo(
            cfg,
            min_image_size=800,
            confidence_threshold=self.confidence_threshold,
            show_mask_heatmaps=False
        )
    
    def detect_and_classify(self, image: np.ndarray) -> Tuple[Dict, float]:
        """
        Detect and classify objects in the image using GLIP
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Tuple of (detection/classification results, inference time)
        """
        # Start timing
        start_time = time.time()
        
        # Convert RGB to BGR for GLIP (if necessary)
        if image.shape[2] == 3:  # RGB format
            image_bgr = image[:, :, ::-1].copy()  # RGB to BGR
        else:
            image_bgr = image
            
        # Create caption for all vehicle classes
        caption = "car van truck bus emergency vehicle"
        
        # Run GLIP detection
        _, top_predictions = self.model.run_on_web_image(image_bgr, caption, self.confidence_threshold)
        
        # Process results
        detections = []
        
        # Check if there are any predictions
        if top_predictions is not None and len(top_predictions) > 0:
            # Get bounding boxes and labels
            boxes = top_predictions.bbox.tolist()
            labels = top_predictions.get_field("labels").tolist()
            scores = top_predictions.get_field("scores").tolist()
            
            # Get entity names from predicted labels
            entities = self.model.entities
            
            # Process each detection
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                # Get entity name (class)
                entity_idx = label - 1  # GLIP labels start from 1
                if entity_idx < len(entities):
                    entity = entities[entity_idx]
                else:
                    continue  # Skip if entity index is out of range
                
                # Map GLIP entity to our vehicle class
                vehicle_class = self._map_to_vehicle_class(entity)
                
                # Only add if it's a vehicle class
                if vehicle_class != "non-vehicle":
                    detections.append({
                        'bbox': box,  # [x1, y1, x2, y2]
                        'class': vehicle_class,
                        'score': score,
                        'area': (box[2] - box[0]) * (box[3] - box[1]),
                    })
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Find main vehicle
        main_vehicle = self.find_main_vehicle(detections, image.shape)
        
        return {
            'detections': detections,
            'main_vehicle': main_vehicle,
        }, inference_time
    
    def _map_to_vehicle_class(self, entity: str) -> str:
        """
        Map GLIP entity to our vehicle class
        
        Args:
            entity: GLIP entity name
            
        Returns:
            Mapped vehicle class
        """
        entity = entity.lower()
        
        # Map to our vehicle classes
        if entity in ["car", "sedan", "coupe", "hatchback", "convertible", "wagon"]:
            return "car"
        elif entity in ["van", "minivan"]:
            return "van"
        elif entity in ["truck", "pickup", "semi", "trailer", "lorry"]:
            return "truck"
        elif entity in ["bus", "coach", "minibus"]:
            return "bus"
        elif entity in ["ambulance", "police", "fire truck", "emergency"]:
            return "emergency"
        else:
            return "non-vehicle"
    
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