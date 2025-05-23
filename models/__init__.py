"""
Models package initialization
"""
import logging

# Configure logging for the models package
logger = logging.getLogger(__name__)

# Import factory classes for easy access
from models.detectors.detector_factory import DetectorFactory
from models.classifiers.classifier_factory import ClassifierFactory
from models.end_to_end.end_to_end_factory import EndToEndFactory

# Import base classes
from models.detectors.base_detector import BaseDetector, DetectionResult
from models.classifiers.base_classifier import BaseClassifier, ClassificationResult
from models.end_to_end.base_end_to_end import BaseEndToEndModel, EndToEndDetection, EndToEndResult

__all__ = [
    # Factory classes
    'DetectorFactory',
    'ClassifierFactory', 
    'EndToEndFactory',
    
    # Base classes
    'BaseDetector',
    'BaseClassifier',
    'BaseEndToEndModel',
    
    # Result classes
    'DetectionResult',
    'ClassificationResult',
    'EndToEndDetection',
    'EndToEndResult',
]

# Log successful import
logger.debug("Models package initialized successfully")