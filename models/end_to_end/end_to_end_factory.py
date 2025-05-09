"""
Factory for creating zero-shot object detection models
"""
from typing import Optional, Dict, Any

from models.end_to_end.yolo_world import YOLOWorldDetector
from models.end_to_end.owlv2 import OWLv2Detector
from models.end_to_end.grounding_dino import GroundingDINODetector


class EndToEndFactory:
    """Factory for creating zero-shot object detection models"""
    
    DETECTOR_TYPES = {
        'yolo_world': YOLOWorldDetector,
        'owlv2': OWLv2Detector,
        'dino': GroundingDINODetector
    }
    
    @classmethod
    def create_detector(cls, 
                      detector_type: str, 
                      model_size: str = None,
                      confidence_threshold: float = None,
                      custom_model_path: Optional[str] = None,
                      **kwargs) -> Any:
        """
        Create a detector of the specified type
        
        Args:
            detector_type: Type of detector to create (yolo_world, owlv2, grounding_dino)
            model_size: Size of the model (varies by implementation)
            confidence_threshold: Confidence threshold for detections
            custom_model_path: Path to custom model weights
            **kwargs: Additional arguments to pass to the detector constructor
            
        Returns:
            Initialized detector
        """
        detector_type = detector_type.lower()
        
        if detector_type not in cls.DETECTOR_TYPES:
            raise ValueError(f"Invalid detector type: {detector_type}. "
                          f"Choose from {list(cls.DETECTOR_TYPES.keys())}")
        
        # Set default model size based on detector type if not provided
        if model_size is None:
            default_sizes = {
                'yolo_world': 'medium',
                'owlv2': 'base',
                'dino': 'base'
            }
            model_size = default_sizes[detector_type]
        
        # Create detector
        detector_class = cls.DETECTOR_TYPES[detector_type]
        
        return detector_class(
            model_size=model_size,
            confidence_threshold=confidence_threshold,
            custom_model_path=custom_model_path,
            **kwargs
        )
    
    @classmethod
    def get_available_detector_types(cls) -> list:
        """Get a list of available detector types"""
        return list(cls.DETECTOR_TYPES.keys())