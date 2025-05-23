"""
CLIP classifier implementation for zero-shot image classification
"""
import logging
from typing import List

import numpy as np
import torch
import clip
from PIL import Image

from .base_classifier import BaseClassifier, ClassificationResult
from config import VEHICLE_CLASSES

logger = logging.getLogger(__name__)


class ClipClassifier(BaseClassifier):
    """CLIP-based zero-shot classifier"""
    
    def _initialize_model(self) -> None:
        """Initialize CLIP model"""
        try:
            # Get model configuration
            model_sizes = self.config.get('model_sizes', {})
            
            if self.model_size not in model_sizes:
                raise ValueError(f"Invalid model size '{self.model_size}'. "
                               f"Available sizes: {list(model_sizes.keys())}")
            
            clip_model_name = model_sizes[self.model_size]
            
            # Load CLIP model and preprocess function
            self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
            
            # Pre-compute text embeddings for efficiency
            self.text_features = self._compute_text_embeddings()
            
            logger.info(f"Successfully initialized CLIP model '{clip_model_name}' on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise
    
    def _compute_text_embeddings(self) -> torch.Tensor:
        """
        Pre-compute text embeddings for all vehicle classes
        
        Returns:
            Tensor of text embeddings for each class
        """
        try:
            all_prompts = []
            class_to_indices = []
            
            # Generate prompts for each class
            for cls in VEHICLE_CLASSES:
                cls_prompts = self.text_prompts[cls]
                
                # Store indices for this class
                start_idx = len(all_prompts)
                all_prompts.extend(cls_prompts)
                end_idx = len(all_prompts)
                class_to_indices.append((start_idx, end_idx))
            
            # Tokenize all prompts
            text_inputs = torch.cat([clip.tokenize(prompt) for prompt in all_prompts]).to(self.device)
            
            # Encode text features
            with torch.no_grad():
                text_features_all = self.model.encode_text(text_inputs)
                text_features_all = text_features_all / text_features_all.norm(dim=-1, keepdim=True)
                
                # Average embeddings for each class
                text_features = []
                for start, end in class_to_indices:
                    class_features = text_features_all[start:end]
                    mean_feature = class_features.mean(dim=0)
                    mean_feature = mean_feature / mean_feature.norm()  # Normalize again
                    text_features.append(mean_feature)
                
                class_text_features = torch.stack(text_features)
            
            logger.debug(f"Pre-computed text embeddings for {len(VEHICLE_CLASSES)} classes")
            return class_text_features
            
        except Exception as e:
            logger.error(f"Failed to compute text embeddings: {e}")
            raise
    
    def _run_inference(self, image: Image.Image) -> ClassificationResult:
        """
        Run CLIP inference on the image
        
        Args:
            image: Input image as PIL Image
            
        Returns:
            ClassificationResult object
        """
        try:
            # Preprocess image for CLIP
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Get image features
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                scores = similarity[0].cpu().numpy()
                
                # Create scores dictionary
                class_scores = {cls: float(score) for cls, score in zip(VEHICLE_CLASSES, scores)}
                
                # Get the class with the highest score
                top_class = max(class_scores.items(), key=lambda x: x[1])[0]
                top_score = class_scores[top_class]
            
            return ClassificationResult(
                predicted_class=top_class,
                confidence=top_score,
                all_scores=class_scores
            )
            
        except Exception as e:
            logger.error(f"CLIP inference failed: {e}")
            raise