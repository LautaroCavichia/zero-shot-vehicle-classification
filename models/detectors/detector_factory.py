"""
Detector factory for creating different types of detectors
"""
import logging
from typing import Dict, Any, Optional, Type

from .yolo import Yolov12Detector
from .supervision_detector import SupervisionDetector
from .ssd import SsdDetector
from .base_detector import BaseDetector

logger = logging.getLogger(__name__)


class DetectorFactory:
    """Factory class for creating different types of detectors"""
    
    # Registry of available detectors
    _DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {
        "yolov12": Yolov12Detector,
        "supervision": SupervisionDetector,
        "ssd": SsdDetector,
    }
    
    @classmethod
    def create_detector(
        cls,
        detector_type: str,
        model_size: str = None,
        confidence_threshold: float = None,
        custom_model_path: Optional[str] = None,
        **kwargs
    ) -> BaseDetector:
        """
        Create a detector of the specified type
        
        Args:
            detector_type: Type of detector to create
            model_size: Size of the model to use
            confidence_threshold: Confidence threshold for detections
            custom_model_path: Path to custom model weights
            **kwargs: Additional arguments to pass to the detector constructor
            
        Returns:
            Detector instance
            
        Raises:
            ValueError: If detector type is not supported
        """
        detector_type = detector_type.lower()
        
        if detector_type not in cls._DETECTOR_REGISTRY:
            available_types = list(cls._DETECTOR_REGISTRY.keys())
            raise ValueError(f"Invalid detector type: '{detector_type}'. "
                           f"Available types: {available_types}")
        
        detector_class = cls._DETECTOR_REGISTRY[detector_type]
        
        try:
            # Create detector instance
            detector = detector_class(
                model_size=model_size,
                confidence_threshold=confidence_threshold,
                custom_model_path=custom_model_path,
                **kwargs
            )
            
            logger.info(f"Successfully created {detector_type} detector")
            return detector
            
        except Exception as e:
            logger.error(f"Failed to create {detector_type} detector: {e}")
            raise
    
    @classmethod
    def get_available_detectors(cls) -> list:
        """Get a list of available detector types"""
        return list(cls._DETECTOR_REGISTRY.keys())
    
    @classmethod
    def register_detector(cls, name: str, detector_class: Type[BaseDetector]) -> None:
        """
        Register a new detector type
        
        Args:
            name: Name of the detector type
            detector_class: Detector class that inherits from BaseDetector
        """
        if not issubclass(detector_class, BaseDetector):
            raise ValueError(f"Detector class must inherit from BaseDetector")
        
        cls._DETECTOR_REGISTRY[name.lower()] = detector_class
        logger.info(f"Registered new detector type: {name}")
    
    @classmethod
    def is_detector_available(cls, detector_type: str) -> bool:
        """
        Check if a detector type is available
        
        Args:
            detector_type: Type of detector to check
            
        Returns:
            True if detector is available, False otherwise
        """
        return detector_type.lower() in cls._DETECTOR_REGISTRY