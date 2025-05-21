"""
Factory for creating classifier instances
"""
from typing import Dict, Any, Optional, Union, Type
from models.classifiers.clip import CLIPClassifier
from models.classifiers.openclip import OpenCLIPClassifier
from models.classifiers.git import GitClassifier
from models.classifiers.siglip import SigLIPClassifier

class ClassifierFactory:
    """Factory for creating classifier instances"""
    # Map classifier names to their corresponding classes
    CLASSIFIER_MAP = {
        'clip': CLIPClassifier,
        'openclip': OpenCLIPClassifier,
        'git': GitClassifier,
        'siglip': SigLIPClassifier
    }
    
    @classmethod
    def create_classifier(cls, classifier_type: str, **kwargs):
        """
        Create a classifier instance of the specified type
        
        Args:
            classifier_type: Type of classifier to create (one of 'clip', 'openclip', 'git', 'siglip')
            **kwargs: Additional arguments to pass to the classifier constructor
            
        Returns:
            Classifier instance
            
        Raises:
            ValueError: If the classifier type is not supported
        """
        classifier_type = classifier_type.lower()
        if classifier_type not in cls.CLASSIFIER_MAP:
            raise ValueError(f"Unsupported classifier type: {classifier_type}. "
                           f"Choose from {list(cls.CLASSIFIER_MAP.keys())}")
        
        # Get the classifier class
        classifier_class = cls.CLASSIFIER_MAP[classifier_type]
        
        # Create and return an instance of the classifier
        return classifier_class(**kwargs)