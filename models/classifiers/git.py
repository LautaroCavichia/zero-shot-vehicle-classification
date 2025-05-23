"""
GIT (Generative Image-to-Text) classifier implementation for zero-shot image classification
"""
import logging
from typing import List, Dict

import numpy as np
import warnings
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

from .base_classifier import BaseClassifier, ClassificationResult
from config import VEHICLE_CLASSES

logger = logging.getLogger(__name__)


class GitClassifier(BaseClassifier):
    """GIT-based zero-shot classifier using image captioning and similarity matching"""
    
    def _initialize_model(self) -> None:
        """Initialize GIT model and sentence transformer"""
        try:
            # Get model configuration
            git_model_name = self.config.get('model_name', 'microsoft/git-base')
            similarity_model_name = self.config.get('similarity_model', 'all-MiniLM-L6-v2')
            self.max_caption_length = self.config.get('max_caption_length', 40)
            
            # Initialize GIT model for captioning
            self.git_processor = AutoProcessor.from_pretrained(git_model_name)
            self.git_model = AutoModelForCausalLM.from_pretrained(git_model_name).to(self.device)
            
            # Initialize sentence transformer for similarity
            self.similarity_model = SentenceTransformer(similarity_model_name, device=self.device)
            
            # Prepare class descriptions and embeddings
            self._prepare_class_embeddings()
            
            logger.info(f"Successfully initialized GIT model '{git_model_name}' and "
                       f"similarity model '{similarity_model_name}' on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GIT classifier: {e}")
            raise
    
    def _prepare_class_embeddings(self) -> None:
        """Prepare class descriptions and pre-compute their embeddings"""
        try:
            # Create a flat list of all descriptions for embedding
            self.all_descriptions = []
            self.description_to_class = {}
            
            for cls in VEHICLE_CLASSES:
                cls_descriptions = self.text_prompts[cls]
                self.all_descriptions.extend(cls_descriptions)
                
                for desc in cls_descriptions:
                    self.description_to_class[desc] = cls
            
            # Pre-compute embeddings for all class descriptions
            logger.debug("Pre-computing embeddings for class descriptions...")
            self.description_embeddings = self.similarity_model.encode(
                self.all_descriptions, 
                convert_to_tensor=True, 
                device=self.device
            )
            
            logger.debug(f"Pre-computed {len(self.all_descriptions)} description embeddings")
            
        except Exception as e:
            logger.error(f"Failed to prepare class embeddings: {e}")
            raise
    
    def _generate_caption(self, image_inputs: Dict) -> str:
        """
        Generate a caption for the image using GIT model
        
        Args:
            image_inputs: Processed image inputs for GIT
            
        Returns:
            Generated caption string
        """
        try:
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    generated_ids = self.git_model.generate(
                        pixel_values=image_inputs["pixel_values"],
                        max_length=self.max_caption_length,
                        do_sample=False,
                        pad_token_id=self.git_processor.tokenizer.eos_token_id
                    )
                # Ensure the model is in evaluation mode       
            # Decode the generated caption
            caption = self.git_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            # Basic validation of caption
            if not caption or len(caption.split()) < 2:
                caption = "an image"  # Fallback caption
            
            return caption
            
        except Exception as e:
            logger.error(f"Failed to generate caption: {e}")
            return "an image"  # Fallback
    
    def _run_inference(self, image: Image.Image) -> ClassificationResult:
        """
        Run GIT inference on the image
        
        Args:
            image: Input image as PIL Image
            
        Returns:
            ClassificationResult object
        """
        try:
            # Process image for GIT
            image_inputs = self.git_processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate caption using GIT
            caption = self._generate_caption(image_inputs)
            logger.debug(f"Generated caption: '{caption}'")
            
            # Embed the generated caption
            caption_embedding = self.similarity_model.encode(
                caption, 
                convert_to_tensor=True, 
                device=self.device
            )
            
            # Calculate cosine similarity between caption and all pre-computed description embeddings
            cosine_scores = util.pytorch_cos_sim(caption_embedding, self.description_embeddings)[0]
            cosine_scores = cosine_scores.cpu()
            
            # Aggregate scores per class (take maximum similarity for any description in that class)
            class_scores = {cls: 0.0 for cls in VEHICLE_CLASSES}
            for i, description in enumerate(self.all_descriptions):
                cls = self.description_to_class[description]
                # Update class score with the highest similarity found so far for that class
                class_scores[cls] = max(class_scores[cls], cosine_scores[i].item())
            
            # Normalize scores using softmax for probabilities
            valid_classes = [cls for cls, score in class_scores.items() if score > 0]
            
            if not valid_classes:
                # Handle case where no description had positive similarity (unlikely)
                normalized_class_scores = {cls: 1.0/len(VEHICLE_CLASSES) for cls in VEHICLE_CLASSES}
                top_class = "non-vehicle"  # Default fallback
                top_score = normalized_class_scores[top_class]
            else:
                scores_tensor = torch.tensor([class_scores[cls] for cls in valid_classes])
                normalized_scores = torch.nn.functional.softmax(scores_tensor, dim=0).tolist()
                
                normalized_class_scores = {cls: 0.0 for cls in VEHICLE_CLASSES}
                for i, cls in enumerate(valid_classes):
                    normalized_class_scores[cls] = normalized_scores[i]
                
                top_class = max(normalized_class_scores.items(), key=lambda x: x[1])[0]
                top_score = normalized_class_scores[top_class]
            
            return ClassificationResult(
                predicted_class=top_class,
                confidence=top_score,
                all_scores=normalized_class_scores
            )
            
        except Exception as e:
            logger.error(f"GIT inference failed: {e}")
            raise