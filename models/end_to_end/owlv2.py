"""
OWLv2 model implementation for zero-shot object detection and classification
"""
import logging
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from .base_end_to_end import BaseEndToEndModel, EndToEndDetection
from config import VEHICLE_CLASSES

logger = logging.getLogger(__name__)


class Owlv2Detector(BaseEndToEndModel):
    """OWLv2 detector for zero-shot object detection and classification"""
    
    def _initialize_model(self) -> None:
        """Initialize OWLv2 model"""
        try:
            # Get model configuration
            model_sizes = self.config.get('model_sizes', {})
            
            if self.custom_model_path:
                model_path = self.custom_model_path
                logger.info(f"Using custom model path: {model_path}")
            elif self.model_size in model_sizes:
                model_path = model_sizes[self.model_size]
                logger.info(f"Using model: {model_path}")
            else:
                raise ValueError(f"Invalid model size '{self.model_size}'. "
                               f"Available sizes: {list(model_sizes.keys())}")
            
            # Initialize OWLv2 processor and model
            self.processor = Owlv2Processor.from_pretrained(model_path)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_path)
            self.model = self.model.to(self.device)
            
            # Prepare text queries for vehicle classes
            self.text_queries = self._prepare_text_queries()
            
            logger.info(f"Successfully initialized OWLv2 model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OWLv2 model: {e}")
            raise
    
    def _prepare_text_queries(self) -> List[List[str]]:
        """
        Prepare text queries for vehicle detection
        
        Returns:
            List of text queries formatted for OWLv2
        """
        # Convert vehicle classes to prompt format expected by OWLv2
        text_queries = [[f"a photo of a {cls}" for cls in VEHICLE_CLASSES]]
        logger.debug(f"Prepared text queries: {text_queries[0]}")
        return text_queries
    
    def _run_inference(self, image: Image.Image) -> List[EndToEndDetection]:
        """
        Run OWLv2 inference on the image
        
        Args:
            image: Input image as PIL Image
            
        Returns:
            List of EndToEndDetection objects
        """
        try:
            # Process inputs
            inputs = self.processor(
                text=self.text_queries,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run model inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Target image sizes for rescaling boxes
            target_sizes = torch.tensor([(image.height, image.width)]).to(self.device)
            
            # Post-process results
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold,
                text_labels=self.text_queries
            )
            
            # Extract results for the first (and only) image
            result = results[0]
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["text_labels"]
            
            # Convert to CPU and numpy for processing
            boxes_np = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
            scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
            
            detections = []
            
            # Process each detection
            for i, (box, score, label) in enumerate(zip(boxes_np, scores_np, labels)):
                # Extract class name from the prompt format "a photo of a {class}"
                class_name = label.replace("a photo of a ", "")
                
                # Map to standardized vehicle classes
                mapped_class = self._map_to_vehicle_class(class_name)
                
                # Get box coordinates (already in x1, y1, x2, y2 format)
                x1, y1, x2, y2 = box.tolist() if hasattr(box, 'tolist') else box
                
                # Calculate area
                area = (x2 - x1) * (y2 - y1)
                
                # Create detection
                detection = EndToEndDetection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    predicted_class=mapped_class,
                    confidence=float(score),
                    area=float(area)
                )
                
                detections.append(detection)
            
            logger.debug(f"OWLv2 detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"OWLv2 inference failed: {e}")
            return []