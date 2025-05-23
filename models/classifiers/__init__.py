"""
Classifiers package initialization
"""
from .base_classifier import BaseClassifier, ClassificationResult
from .clip import ClipClassifier
from .openclip import OpenclipClassifier
from .siglip import SiglipClassifier
from .git import GitClassifier
from .classifier_factory import ClassifierFactory

__all__ = [
    'BaseClassifier',
    'ClassificationResult',
    'ClipClassifier',
    'OpenclipClassifier',
    'SiglipClassifier',
    'GitClassifier',
    'ClassifierFactory',
]