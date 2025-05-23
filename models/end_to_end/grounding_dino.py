"""
GroundingDINO model implementation for zero-shot object detection and classification
"""
import logging
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from .base_end_to_end import BaseEndToEndModel, EndToEndDetection
from config import VEHICLE_CLASSES

logger = logging.getLogger(__name__)


class GroundingDinoDetector(BaseEndToEndModel):
    """GroundingDINO detector for zero-shot object detection and classification"""
    
    def _initialize_model(self) -> None:
        """Initialize GroundingDINO model"""
        try:
            # Get model configuration
            model_sizes = self.config.get('model_sizes', {})
            self.text_threshold = self.config.get('text_threshold', 0.25)
            
            if self.custom_model_path:
                model_path = self.custom_model_path
                logger.info(f"Using custom model path: {model_path}")
            elif self.model_size in model_sizes:
                model_path = model_sizes[self.model_size]
                logger.info(f"Using model: {model_path}")
            else:
                raise ValueError(f"Invalid model size '{self.model_size}'. "
                               f"Available sizes: {list(model_sizes.keys())}")
            
            # Initialize GroundingDINO processor and model
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device)
            
            # Prepare text prompts for vehicle detection
            self.text_prompts = self._prepare_text_prompts()
            
            logger.info(f"Successfully initialized GroundingDINO model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GroundingDINO model: {e}")
            raise
    
    def _prepare_text_prompts(self) -> List[List[str]]:
        """
        Prepare text prompts for vehicle detection
        
        Returns:
            List of text prompts formatted for GroundingDINO
        """
        # Create text prompts for GroundingDINO (expects list of lists for batch processing)
        text_prompts = [VEHICLE_CLASSES]
        logger.debug(f"Prepared text prompts: {text_prompts[0]}")
        return text_prompts
    
    def _run_inference(self, image: Image.Image) -> List[EndToEndDetection]:
        """
        Run GroundingDINO inference on the image
        
        Args:
            image: Input image as PIL Image
            
        Returns:
            List of EndToEndDetection objects
        """
        try:
            # Prepare inputs
            inputs = self.processor(
                images=image, 
                text=self.text_prompts, 
                return_tensors="pt"
            ).to(self.device)
            
            # Run model inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            results_list = self.processor.post_process_grounded_object_detection(
                outputs,
                box_threshold=self.confidence_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]]  # [height, width]
            )
            
            # Check if results are valid
            if not results_list or not isinstance(results_list, list) or not results_list:
                logger.warning("GroundingDINO returned no valid results")
                return []
            
            result = results_list[0]  # First batch result
            
            detections = []
            
            # Ensure all necessary keys exist in the result
            boxes_tensor = result.get("boxes", torch.empty(0))
            scores_tensor = result.get("scores", torch.empty(0))
            text_labels_list = result.get("text_labels", [])
            
            # Validate result consistency
            if not (len(boxes_tensor) == len(scores_tensor) == len(text_labels_list)):
                logger.warning(f"Mismatch in GroundingDINO result lengths: "
                              f"boxes ({len(boxes_tensor)}), scores ({len(scores_tensor)}), "
                              f"text_labels ({len(text_labels_list)})")
                return []
            
            # Process each detection
            for box_coords, score_val, text_label_str in zip(boxes_tensor, scores_tensor, text_labels_list):
                # Get box coordinates [x1, y1, x2, y2]
                x1, y1, x2, y2 = box_coords.tolist()
                
                # Map to standardized vehicle classes
                mapped_class = self._map_to_vehicle_class(text_label_str)
                
                # Calculate area
                area = (x2 - x1) * (y2 - y1)
                
                # Create detection
                detection = EndToEndDetection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    predicted_class=mapped_class,
                    confidence=float(score_val.item()),
                    area=float(area)
                )
                
                detections.append(detection)
            
            logger.debug(f"GroundingDINO detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"GroundingDINO inference failed: {e}")
            return []