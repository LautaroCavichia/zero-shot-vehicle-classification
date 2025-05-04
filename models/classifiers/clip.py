"""
CLIP classifier implementation for zero-shot image classification
"""
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import clip
from PIL import Image

from config import CLASSIFIER_CONFIGS, CLIP_TEMPLATES, CLASS_DESCRIPTIONS, VEHICLE_CLASSES


class CLIPClassifier:
    """CLIP-based zero-shot classifier"""
    
    MODEL_SIZES = {
        "small": "ViT-B/16",
        "medium": "ViT-B/32",
        "large": "ViT-L/14"
    }
    
    def __init__(self, model_size: str = "medium"):
        """
        Initialize CLIP classifier
        
        Args:
            model_size: Name of the CLIP model to use
        """
        # Use config values if not provided
        config = CLASSIFIER_CONFIGS['clip']
        self._model_size = model_size.lower()
        
        # Set up model path
        if self._model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model size: {model_size}. "
                          f"Choose from {list(self.MODEL_SIZES.keys())}")
        
        self.clip_model = self.MODEL_SIZES[self._model_size]
        
        # Device configuration
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CLIP classifier using device: {self.device}")
        
        # Load CLIP model and preprocess
        self.model, self.preprocess = clip.load(self.clip_model, device=self.device)
        
        # Pre-compute text embeddings for each class
        self.text_features = self._compute_text_embeddings()
        
    def _compute_text_embeddings(self) -> torch.Tensor:
        """
        Compute text embeddings for each class using various templates and descriptions
        
        Returns:
            Tensor of text embeddings for each class
        """
        all_prompts = []
        class_to_indices = []
        
        # Generate prompts for each class
        for cls in VEHICLE_CLASSES:
            # Add basic templates
            prompts = [template.format(cls) for template in CLIP_TEMPLATES]
            
            # Add class-specific descriptions
            if cls in CLASS_DESCRIPTIONS:
                prompts.extend(CLASS_DESCRIPTIONS[cls])
            
            # Store indices for this class
            start_idx = len(all_prompts)
            all_prompts.extend(prompts)
            end_idx = len(all_prompts)
            class_to_indices.append((start_idx, end_idx))
        
        # Tokenize all prompts
        text_inputs = torch.cat([clip.tokenize(p) for p in all_prompts]).to(self.device)
        
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
        
        return class_text_features
        
    def classify(self, image: np.ndarray) -> Tuple[Dict, float]:
        """
        Classify an image using CLIP
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Tuple of (classification result dict, inference time)
        """
        # Start timing
        start_time = time.time()
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # Check if the image has valid dimensions
        if image.size[0] == 0 or image.size[1] == 0:
            # Return a default result if image is invalid
            end_time = time.time()
            return {
                'class': 'non-vehicle',
                'score': 0.9,
                'all_scores': {cls: 0.0 for cls in VEHICLE_CLASSES}
            }, end_time - start_time
        
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
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        return {
            'class': top_class,
            'score': top_score,
            'all_scores': class_scores,
        }, inference_time