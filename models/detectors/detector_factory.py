"""
Detector factory for creating different types of detectors
"""
from typing import Dict, Any, Optional, List
from models.detectors.supervision_detector import SupervisionDetector
from models.detectors.ssd import SSDDetector
from models.detectors.yolo import YOLOv12Detector

class DetectorFactory:
    """Factory class for creating different types of detectors"""
    
    @staticmethod
    def create_detector(
        detector_type: str,
        **kwargs
    ):
        """
        Create a detector of the specified type
        
        Args:
            detector_type: Type of detector to create ("supervision", "ssd", "yolov12")
            **kwargs: Additional arguments to pass to the detector constructor
            
        Returns:
            Detector instance
        """
        detector_map = {
            "supervision": SupervisionDetector,
            "ssd": SSDDetector,
            "yolov12": YOLOv12Detector,
        }
        
        if detector_type not in detector_map:
            raise ValueError(f"Invalid detector type: {detector_type}. "
                           f"Choose from {list(detector_map.keys())}")
            
        # Create and return detector
        return detector_map[detector_type](**kwargs)