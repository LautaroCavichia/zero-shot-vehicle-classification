"""
Factory for creating classifier instances
"""
import logging
from typing import Dict, Any, Optional, Type

from .clip import ClipClassifier
from .openclip import OpenclipClassifier
from .siglip import SiglipClassifier
from .git import GitClassifier
from .base_classifier import BaseClassifier

logger = logging.getLogger(__name__)


class ClassifierFactory:
    """Factory class for creating classifier instances"""
    
    # Registry of available classifiers
    _CLASSIFIER_REGISTRY: Dict[str, Type[BaseClassifier]] = {
        'clip': ClipClassifier,
        'openclip': OpenclipClassifier,
        'siglip': SiglipClassifier,
        'git': GitClassifier,
    }
    
    @classmethod
    def create_classifier(
        cls, 
        classifier_type: str, 
        model_size: str = None,
        **kwargs
    ) -> BaseClassifier:
        """
        Create a classifier instance of the specified type
        
        Args:
            classifier_type: Type of classifier to create
            model_size: Size of the model to use
            **kwargs: Additional arguments to pass to the classifier constructor
            
        Returns:
            Classifier instance
            
        Raises:
            ValueError: If the classifier type is not supported
        """
        classifier_type = classifier_type.lower()
        
        if classifier_type not in cls._CLASSIFIER_REGISTRY:
            available_types = list(cls._CLASSIFIER_REGISTRY.keys())
            raise ValueError(f"Invalid classifier type: '{classifier_type}'. "
                           f"Available types: {available_types}")
        
        classifier_class = cls._CLASSIFIER_REGISTRY[classifier_type]
        
        try:
            # Create classifier instance
            classifier = classifier_class(
                model_size=model_size,
                **kwargs
            )
            
            logger.info(f"Successfully created {classifier_type} classifier")
            return classifier
            
        except Exception as e:
            logger.error(f"Failed to create {classifier_type} classifier: {e}")
            raise
    
    @classmethod
    def get_available_classifiers(cls) -> list:
        """Get a list of available classifier types"""
        return list(cls._CLASSIFIER_REGISTRY.keys())
    
    @classmethod
    def register_classifier(cls, name: str, classifier_class: Type[BaseClassifier]) -> None:
        """
        Register a new classifier type
        
        Args:
            name: Name of the classifier type
            classifier_class: Classifier class that inherits from BaseClassifier
        """
        if not issubclass(classifier_class, BaseClassifier):
            raise ValueError(f"Classifier class must inherit from BaseClassifier")
        
        cls._CLASSIFIER_REGISTRY[name.lower()] = classifier_class
        logger.info(f"Registered new classifier type: {name}")
    
    @classmethod
    def is_classifier_available(cls, classifier_type: str) -> bool:
        """
        Check if a classifier type is available
        
        Args:
            classifier_type: Type of classifier to check
            
        Returns:
            True if classifier is available, False otherwise
        """
        return classifier_type.lower() in cls._CLASSIFIER_REGISTRY