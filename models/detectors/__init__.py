"""
Detectors package initialization
"""
from .base_detector import BaseDetector, DetectionResult
from .yolo import Yolov12Detector
from .supervision_detector import SupervisionDetector
from .ssd import SsdDetector
from .detector_factory import DetectorFactory

__all__ = [
    'BaseDetector',
    'DetectionResult',
    'Yolov12Detector',
    'SupervisionDetector', 
    'SsdDetector',
    'DetectorFactory',
]