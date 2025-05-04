"""
Classifiers package initialization
"""
from models.classifiers.clip import CLIPClassifier
from models.classifiers.openclip import OpenCLIPClassifier
from models.classifiers.vilt import ViLTClassifier
from models.classifiers.git import GitClassifier

__all__ = [
    'CLIPClassifier',
    'OpenCLIPClassifier',
    'ViLTClassifier',
    'GitClassifier',
]