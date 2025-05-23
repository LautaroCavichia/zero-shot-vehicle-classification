"""
Base classifier class for zero-shot image classification models
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import time
import logging

import numpy as np
import torch
from PIL import Image

from config import CLASSIFIER_CONFIGS, VEHICLE_CLASSES, CLIP_TEMPLATES, CLASS_DESCRIPTIONS

logger = logging.getLogger(__name__)


class ClassificationResult:
    """Standardized classification result container"""
    
    def __init__(self, predicted_class: str, confidence: float, all_scores: Dict[str, float]):
        """
        Initialize classification result
        
        Args:
            predicted_class: The predicted vehicle class
            confidence: Confidence score for the prediction
            all_scores: Dictionary of scores for all classes
        """
        self.predicted_class = self._validate_class(predicted_class)
        self.confidence = max(0.0, min(1.0, float(confidence)))
        self.all_scores = self._validate_scores(all_scores)
    
    def _validate_class(self, predicted_class: str) -> str:
        """Validate that the predicted class is in VEHICLE_CLASSES"""
        if predicted_class not in VEHICLE_CLASSES:
            logger.warning(f"Invalid class '{predicted_class}', defaulting to 'non-vehicle'")
            return "non-vehicle"
        return predicted_class
    
    def _validate_scores(self, all_scores: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize class scores"""
        validated_scores = {}
        
        for cls in VEHICLE_CLASSES:
            score = all_scores.get(cls, 0.0)
            validated_scores[cls] = max(0.0, min(1.0, float(score)))
        
        return validated_scores
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'class': self.predicted_class,
            'score': self.confidence,
            'all_scores': self.all_scores
        }


class BaseClassifier(ABC):
    """Abstract base class for zero-shot image classifiers"""
    
    def __init__(self, model_size: str = None):
        """
        Initialize base classifier
        
        Args:
            model_size: Size of the model to use
        """
        self.classifier_name = self.__class__.__name__.replace('Classifier', '').lower()
        self.config = CLASSIFIER_CONFIGS.get(self.classifier_name, {})
        
        # Set model parameters
        self.model_size = model_size or self.config.get('default_model_size', 'medium')
        
        # Set up device
        self.device = self._get_best_device()
        
        # Prepare text prompts BEFORE model initialization
        self.text_prompts = self._prepare_text_prompts()
        
        # Initialize model (to be implemented by subclasses)
        self.model = None
        self._initialize_model()
        
        logger.info(f"Initialized {self.classifier_name} classifier with model size: {self.model_size}, "
                   f"device: {self.device}")
    
    def _get_best_device(self) -> str:
        """Get the best available device based on configuration preferences"""
        device_preferences = self.config.get('device_preference', ['cuda', 'mps', 'cpu'])
        
        for device in device_preferences:
            if device == 'cuda' and torch.cuda.is_available():
                return 'cuda'
            elif device == 'mps' and torch.backends.mps.is_available():
                return 'mps'
            elif device == 'cpu':
                return 'cpu'
        
        return 'cpu'  # Fallback
    
    def _prepare_text_prompts(self) -> Dict[str, List[str]]:
        """
        Prepare text prompts for each vehicle class
        
        Returns:
            Dictionary mapping class names to lists of text prompts
        """
        prompts = {}
        
        for cls in VEHICLE_CLASSES:
            cls_prompts = []
            
            # Add template-based prompts
            for template in CLIP_TEMPLATES:
                cls_prompts.append(template.format(cls))
            
            # Add class-specific descriptions
            if cls in CLASS_DESCRIPTIONS:
                cls_prompts.extend(CLASS_DESCRIPTIONS[cls])
            
            # Remove duplicates while preserving order
            prompts[cls] = list(dict.fromkeys(cls_prompts))
        
        return prompts
    
    def _validate_image(self, image: np.ndarray) -> Image.Image:
        """
        Validate and convert image to PIL format
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            PIL Image object
        """
        if image is None:
            raise ValueError("Image cannot be None")
        
        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Image array is empty")
            
            if image.ndim not in [2, 3]:
                raise ValueError(f"Image must be 2D or 3D array, got {image.ndim}D")
            
            # Convert to PIL Image
            if image.ndim == 2:  # Grayscale
                pil_image = Image.fromarray(image, mode='L').convert('RGB')
            else:  # RGB
                pil_image = Image.fromarray(image.astype('uint8'), mode='RGB')
            
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Check image dimensions
        if pil_image.size[0] == 0 or pil_image.size[1] == 0:
            raise ValueError(f"Image has invalid dimensions: {pil_image.size}")
        
        return pil_image
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize the classification model (to be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _run_inference(self, image: Image.Image) -> ClassificationResult:
        """
        Run model inference on the image
        
        Args:
            image: Input image as PIL Image
            
        Returns:
            ClassificationResult object
        """
        pass
    
    def classify(self, image: np.ndarray) -> Tuple[Dict, float]:
        """
        Classify an image using the zero-shot classifier
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Tuple of (classification result dict, inference time)
        """
        start_time = time.time()
        
        try:
            # Validate and convert image
            pil_image = self._validate_image(image)
            
            # Run model-specific inference
            result = self._run_inference(pil_image)
            
            inference_time = time.time() - start_time
            
            logger.debug(f"Classified image as '{result.predicted_class}' with confidence "
                        f"{result.confidence:.3f} in {inference_time:.3f}s")
            
            return result.to_dict(), inference_time
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            
            # Return safe fallback
            fallback_result = ClassificationResult(
                predicted_class="non-vehicle",
                confidence=0.0,
                all_scores={cls: 0.0 for cls in VEHICLE_CLASSES}
            )
            
            return fallback_result.to_dict(), time.time() - start_time