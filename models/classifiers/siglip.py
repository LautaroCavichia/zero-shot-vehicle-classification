"""
SigLIP classifier implementation for zero-shot image classification
Improved version based on official documentation while maintaining compatibility
"""
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

from config import CLASSIFIER_CONFIGS, CLIP_TEMPLATES, CLASS_DESCRIPTIONS, VEHICLE_CLASSES

class SigLIPClassifier:
    """SigLIP-based zero-shot classifier"""
    
    MODEL_SIZES = {
        "medium": "google/siglip2-base-patch16-naflex",
        "large": "google/siglip-so400m-patch14-384"
    }
    
    def __init__(self, model_size: str = "medium"):
        """
        Initialize SigLIP classifier
        
        Args:
            model_size: Size of the model to use (medium, large)
        """
        # Use config values if not provided
        config = CLASSIFIER_CONFIGS.get('siglip', {'model_size': 'medium'})
        self._model_size = model_size.lower()
        
        # Set up model path
        if self._model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model size: {model_size}. "
                          f"Choose from {list(self.MODEL_SIZES.keys())}")
        
        self.model_name = self.MODEL_SIZES[self._model_size]
        
        # Device configuration
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SigLIP classifier using device: {self.device}")
        
        # Load SigLIP model and processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
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
            
            # Add recommended SigLIP template (from documentation)
            prompts.append(f"This is a photo of {cls}.")
            
            # Add class-specific descriptions
            if cls in CLASS_DESCRIPTIONS:
                prompts.extend(CLASS_DESCRIPTIONS[cls])
            
            # Store indices for this class
            start_idx = len(all_prompts)
            all_prompts.extend(prompts)
            end_idx = len(all_prompts)
            class_to_indices.append((start_idx, end_idx))
        
        # Process text inputs
        with torch.no_grad():
            # Process all text prompts - using padding="max_length" as recommended in docs
            try:
                text_inputs = self.processor(
                    text=all_prompts,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
            except Exception as e:
                print(f"Warning: Failed to use max_length padding, falling back to default: {e}")
                text_inputs = self.processor(
                    text=all_prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
            
            # Get text embeddings from SigLIP model
            text_features_all = self.model.get_text_features(**text_inputs)
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
        Classify an image using SigLIP
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Tuple of (classification result dict, inference time)
        """
        # Start timing
        start_time = time.time()
        
        # Add robust error handling while maintaining original behavior
        try:
            # Handle None input
            if image is None:
                print("Warning: Received None image in SigLIP classifier")
                return {
                    'class': 'non-vehicle',
                    'score': 0.9,
                    'all_scores': {cls: 0.0 for cls in VEHICLE_CLASSES}
                }, 0.0
                
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                # Handle empty arrays
                if image.size == 0:
                    print("Warning: Empty image array provided to SigLIP classifier")
                    return {
                        'class': 'non-vehicle',
                        'score': 0.9,
                        'all_scores': {cls: 0.0 for cls in VEHICLE_CLASSES}
                    }, time.time() - start_time
                
                # Convert to PIL
                image = Image.fromarray(image.astype('uint8'))
            
            # Check if the image has valid dimensions
            if image.size[0] == 0 or image.size[1] == 0:
                # Return a default result if image is invalid
                print(f"Warning: Invalid image dimensions: {image.size}")
                return {
                    'class': 'non-vehicle',
                    'score': 0.9,
                    'all_scores': {cls: 0.0 for cls in VEHICLE_CLASSES}
                }, time.time() - start_time
            
            # Process image for SigLIP
            with torch.no_grad():
                # Preprocess image - try to use max_length padding as recommended
                try:
                    image_inputs = self.processor(
                        images=image,
                        padding="max_length",
                        return_tensors="pt"
                    ).to(self.device)
                except Exception as e:
                    # Fallback to original method if max_length fails
                    print(f"Warning: Failed to use max_length padding for image, using default: {e}")
                    image_inputs = self.processor(
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                
                # Get image features
                image_features = self.model.get_image_features(**image_inputs)
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
            
        except Exception as e:
            # Catch-all exception handler
            print(f"Error in SigLIP classification: {e}")
            return {
                'class': 'non-vehicle',
                'score': 0.9,
                'all_scores': {cls: 0.0 for cls in VEHICLE_CLASSES}
            }, time.time() - start_time