"""
OpenCLIP classifier implementation for zero-shot image classification
"""
import logging
from typing import List

import numpy as np
import torch
import open_clip
from PIL import Image

from .base_classifier import BaseClassifier, ClassificationResult
from config import VEHICLE_CLASSES

logger = logging.getLogger(__name__)


class OpenclipClassifier(BaseClassifier):
    """OpenCLIP-based zero-shot classifier"""
    
    def _initialize_model(self) -> None:
        """Initialize OpenCLIP model"""
        try:
            # Get model configuration
            model_sizes = self.config.get('model_sizes', {})
            pretrained_weights = self.config.get('pretrained_weights', {})
            
            if self.model_size not in model_sizes:
                raise ValueError(f"Invalid model size '{self.model_size}'. "
                               f"Available sizes: {list(model_sizes.keys())}")
            
            model_name = model_sizes[self.model_size]
            pretrained = pretrained_weights.get(self.model_size, 'openai')
            
            # Load model, preprocess, and tokenizer
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            
            # Pre-compute text embeddings for efficiency
            self.text_embeddings = self._compute_text_embeddings()
            
            logger.info(f"Successfully initialized OpenCLIP model '{model_name}' with pretrained '{pretrained}' on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenCLIP model: {e}")
            raise
    
    def _compute_text_embeddings(self) -> torch.Tensor:
        """
        Pre-compute text embeddings for all vehicle classes
        
        Returns:
            Tensor of text embeddings
        """
        try:
            all_prompts = []
            self.text_mapping = {}
            idx = 0
            
            # Generate prompts for each class
            for cls_idx, cls in enumerate(VEHICLE_CLASSES):
                cls_prompts = self.text_prompts[cls]
                class_indices = []
                
                for prompt in cls_prompts:
                    all_prompts.append(prompt)
                    class_indices.append(idx)
                    idx += 1
                
                # Store mapping of text indices to class
                self.text_mapping[cls_idx] = class_indices
            
            # Process text inputs
            with torch.no_grad():
                text_tokens = self.tokenizer(all_prompts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)
            
            logger.debug(f"Pre-computed {len(all_prompts)} text embeddings for {len(VEHICLE_CLASSES)} classes")
            return text_embeddings
            
        except Exception as e:
            logger.error(f"Failed to compute text embeddings: {e}")
            raise
    
    def _run_inference(self, image: Image.Image) -> ClassificationResult:
        """
        Run OpenCLIP inference on the image
        
        Args:
            image: Input image as PIL Image
            
        Returns:
            ClassificationResult object
        """
        try:
            # Preprocess image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Get image features
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores with all text prompts
                similarity = (100.0 * image_features @ self.text_embeddings.T).softmax(dim=-1)
                
                # Group scores by class
                class_scores = {cls: 0.0 for cls in VEHICLE_CLASSES}
                
                for cls_idx, cls in enumerate(VEHICLE_CLASSES):
                    # Average scores from all templates for this class
                    indices = self.text_mapping[cls_idx]
                    for idx in indices:
                        class_scores[cls] += similarity[0, idx].item()
                    
                    # Average the scores
                    class_scores[cls] /= len(indices) if indices else 1
                
                # Get the class with the highest score
                top_class = max(class_scores.items(), key=lambda x: x[1])[0]
                top_score = class_scores[top_class]
            
            return ClassificationResult(
                predicted_class=top_class,
                confidence=top_score,
                all_scores=class_scores
            )
            
        except Exception as e:
            logger.error(f"OpenCLIP inference failed: {e}")
            raise