"""
Detectors package initialization
"""
from models.detectors.yolo import YOLOv12Detector
from models.detectors.supervision_detector import SupervisionDetector
from models.detectors.ssd import SSDDetector

__all__ = [
    'YOLOv12Detector',
    'SupervisionDetector',
    'SSDDetector',
]