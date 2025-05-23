"""
Factory for creating zero-shot end-to-end object detection models
"""
import logging
from typing import Optional, Dict, Any, Type

from .yolo_world import YoloWorldDetector
from .owlv2 import Owlv2Detector
from .grounding_dino import GroundingDinoDetector
from .base_end_to_end import BaseEndToEndModel

logger = logging.getLogger(__name__)


class EndToEndFactory:
    """Factory for creating zero-shot end-to-end object detection models"""
    
    # Registry of available end-to-end models
    _MODEL_REGISTRY: Dict[str, Type[BaseEndToEndModel]] = {
        'yolo_world': YoloWorldDetector,
        'owlv2': Owlv2Detector,
        'dino': GroundingDinoDetector,
        'grounding_dino': GroundingDinoDetector,  # Alias for backward compatibility
    }
    
    @classmethod
    def create_detector(
        cls, 
        detector_type: str, 
        model_size: str = None,
        confidence_threshold: float = None,
        custom_model_path: Optional[str] = None,
        **kwargs
    ) -> BaseEndToEndModel:
        """
        Create a detector of the specified type
        
        Args:
            detector_type: Type of detector to create
            model_size: Size of the model
            confidence_threshold: Confidence threshold for detections
            custom_model_path: Path to custom model weights
            **kwargs: Additional arguments to pass to the detector constructor
            
        Returns:
            Initialized end-to-end detector
            
        Raises:
            ValueError: If detector type is not supported
        """
        detector_type = detector_type.lower()
        
        if detector_type not in cls._MODEL_REGISTRY:
            available_types = list(cls._MODEL_REGISTRY.keys())
            raise ValueError(f"Invalid detector type: '{detector_type}'. "
                           f"Available types: {available_types}")
        
        detector_class = cls._MODEL_REGISTRY[detector_type]
        
        try:
            # Create detector instance
            detector = detector_class(
                model_size=model_size,
                confidence_threshold=confidence_threshold,
                custom_model_path=custom_model_path,
                **kwargs
            )
            
            logger.info(f"Successfully created {detector_type} end-to-end detector")
            return detector
            
        except Exception as e:
            logger.error(f"Failed to create {detector_type} end-to-end detector: {e}")
            raise
    
    @classmethod
    def get_available_detector_types(cls) -> list:
        """Get a list of available detector types"""
        # Remove aliases to avoid duplicates
        unique_types = []
        seen_classes = set()
        
        for name, detector_class in cls._MODEL_REGISTRY.items():
            if detector_class not in seen_classes:
                unique_types.append(name)
                seen_classes.add(detector_class)
        
        return unique_types
    
    @classmethod
    def register_detector(cls, name: str, detector_class: Type[BaseEndToEndModel]) -> None:
        """
        Register a new end-to-end detector type
        
        Args:
            name: Name of the detector type
            detector_class: Detector class that inherits from BaseEndToEndModel
        """
        if not issubclass(detector_class, BaseEndToEndModel):
            raise ValueError(f"Detector class must inherit from BaseEndToEndModel")
        
        cls._MODEL_REGISTRY[name.lower()] = detector_class
        logger.info(f"Registered new end-to-end detector type: {name}")
    
    @classmethod
    def is_detector_available(cls, detector_type: str) -> bool:
        """
        Check if a detector type is available
        
        Args:
            detector_type: Type of detector to check
            
        Returns:
            True if detector is available, False otherwise
        """
        return detector_type.lower() in cls._MODEL_REGISTRY