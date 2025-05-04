"""
OpenCLIP classifier implementation for zero-shot image classification
"""
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import open_clip
from PIL import Image

from config import CLASSIFIER_CONFIGS, CLIP_TEMPLATES, CLASS_DESCRIPTIONS, VEHICLE_CLASSES


class OpenCLIPClassifier:
    """OpenCLIP-based zero-shot classifier"""
    
    def __init__(self, model_name: str = None, pretrained: str = None):
        """
        Initialize OpenCLIP classifier
        
        Args:
            model_name: Name of the OpenCLIP model architecture
            pretrained: Name of the pretrained weights to use
        """
        # Use config values if not provided
        config = CLASSIFIER_CONFIGS['openclip']
        self.model_name = model_name or config['model_name']
        self.pretrained = pretrained or config['pretrained']
        
        # Load model and processor
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, 
            pretrained=self.pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        
        # Device configuration
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"OpenCLIP classifier using device: {self.device}")
        
        # Pre-compute text embeddings for each class
        self.text_embeddings = self._compute_text_embeddings()
        
    def _compute_text_embeddings(self) -> torch.Tensor:
        """
        Compute text embeddings for each class using various templates and descriptions
        
        Returns:
            Tensor of text embeddings
        """
        text_inputs = []
        self.text_mapping = {}
        idx = 0
        
        # Use templates for each class
        for cls_idx, cls in enumerate(VEHICLE_CLASSES):
            class_texts = []
            
            # Add basic templates
            for template in CLIP_TEMPLATES:
                text = template.format(cls)
                text_inputs.append(text)
                class_texts.append(idx)
                idx += 1
            
            # Add class-specific descriptions
            if cls in CLASS_DESCRIPTIONS:
                for desc in CLASS_DESCRIPTIONS[cls]:
                    text_inputs.append(desc)
                    class_texts.append(idx)
                    idx += 1
            
            # Store mapping of text indices to class
            self.text_mapping[cls_idx] = class_texts
        
        # Process text inputs
        with torch.no_grad():
            text_tokens = self.tokenizer(text_inputs).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_embeddings
        
    def classify(self, image: np.ndarray) -> Tuple[Dict, float]:
        """
        Classify an image using OpenCLIP
        
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
                # Sum scores from all templates for this class
                indices = self.text_mapping[cls_idx]
                for idx in indices:
                    class_scores[cls] += similarity[0, idx].item()
                
                # Average the scores
                class_scores[cls] /= len(indices) if indices else 1
        
        # Get the class with the highest score
        sorted_scores = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        top_class, top_score = sorted_scores[0]
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        return {
            'class': top_class,
            'score': top_score,
            'all_scores': class_scores,
        }, inference_time