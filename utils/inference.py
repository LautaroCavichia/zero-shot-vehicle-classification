"""
Inference utilities for the zero-shot vehicle benchmark
"""
import time
from typing import Dict, List, Tuple, Any, Callable

import numpy as np

from models.detectors.yolo import YOLOv12Detector
from models.detectors.supervision_detector import SupervisionDetector
from models.detectors.ssd import SSDDetector

from models.classifiers.clip import CLIPClassifier
from models.classifiers.openclip import OpenCLIPClassifier
from models.classifiers.git import GitClassifier

from models.end_to_end.yolo_world import YOLOWorldDetector
from models.end_to_end.owlv2 import OWLv2Detector
from models.end_to_end.grounding_dino import GroundingDINODetector


class InferencePipeline:
    """
    Inference pipeline for running detector+classifier combinations
    or end-to-end models for vehicle detection and classification
    """
    
    def __init__(self, detector_name: str = None, classifier_name: str = None, end_to_end: bool = False):
        """
        Initialize inference pipeline
        
        Args:
            detector_name: Name of the detector to use
            classifier_name: Name of the classifier to use
            end_to_end: Whether to use an end-to-end model
        """
        self.end_to_end = end_to_end
        
        if end_to_end:
            # Load end-to-end model
            if detector_name == "dino":
                self.model = GroundingDINODetector(model_size="base")
            elif detector_name == "owlv2":
                self.model = OWLv2Detector(model_size="base")
            elif detector_name == "yolo_world":
                self.model = YOLOWorldDetector(model_size="medium")
            else:
                raise ValueError(f"Unsupported end-to-end model: {detector_name}")
                
            self.detector_name = detector_name
            self.classifier_name = None
        else:
            # Load detector
            if detector_name == "yolov12":
                self.detector = YOLOv12Detector()
            elif detector_name == "supervision":
                self.detector = SupervisionDetector(model_size="medium")
            elif detector_name == "ssd":
                self.detector = SSDDetector(model_size="medium")
            else:
                raise ValueError(f"Unsupported detector: {detector_name}")
                
            # Load classifier
            if classifier_name == "clip":
                self.classifier = CLIPClassifier(model_size="medium")
            elif classifier_name == "openclip":
                self.classifier = OpenCLIPClassifier()
            elif classifier_name == "git":
                self.classifier = GitClassifier()
            else:
                raise ValueError(f"Unsupported classifier: {classifier_name}")
                
            self.detector_name = detector_name
            self.classifier_name = classifier_name
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image through the pipeline
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Dict with results including timing information
        """
        start_time = time.time()
        
        if self.end_to_end:
            # Run end-to-end model
            results, inference_time = self.model.detect_and_classify(image)
            
            # Extract main vehicle
            main_vehicle = results.get('main_vehicle', None)
            
            end_time = time.time()
            
            # Return results
            return {
                'pipeline': f"{self.detector_name}",
                'detection_time': inference_time,
                'classification_time': 0.0,  # No separate classification step
                'total_time': end_time - start_time,
                'detections': results.get('detections', []),
                'main_vehicle': main_vehicle,
                'main_vehicle_class': main_vehicle['class'] if main_vehicle else None,
                'main_vehicle_confidence': main_vehicle['score'] if main_vehicle else None,
            }
        else:
            # Run detector
            detections, detection_time = self.detector.detect(image)
            
            # Find main vehicle
            main_vehicle = self.detector.find_main_vehicle(detections, image.shape)
            
            if main_vehicle is None:
                # No vehicles detected
                classification_time = 0.0
                main_vehicle_class = None
                class_confidence = None
                all_scores = {}
            else:
                # Crop main vehicle
                cropped_vehicle = self.detector.crop_bbox(image, main_vehicle['bbox'])
                
                # Classify cropped vehicle
                classification, classification_time = self.classifier.classify(cropped_vehicle)
                
                # Extract classification results
                main_vehicle_class = classification['class']
                class_confidence = classification['score']
                all_scores = classification.get('all_scores', {})
            
            end_time = time.time()
            
            # Return results
            return {
                'pipeline': f"{self.detector_name}+{self.classifier_name}",
                'detection_time': detection_time,
                'classification_time': classification_time,
                'total_time': end_time - start_time,
                'detections': detections,
                'main_vehicle': main_vehicle,
                'main_vehicle_class': main_vehicle_class,
                'main_vehicle_confidence': class_confidence,
                'class_scores': all_scores,
            }


def create_pipeline(detector: str, classifier: str = None) -> InferencePipeline:
    """
    Factory function to create a pipeline
    
    Args:
        detector: Name of the detector or end-to-end model
        classifier: Name of the classifier (None for end-to-end models)
        
    Returns:
        InferencePipeline instance
    """
    end_to_end = classifier is None and detector in ["dino", "yolo_world", "owlv2"]
    return InferencePipeline(detector, classifier, end_to_end)