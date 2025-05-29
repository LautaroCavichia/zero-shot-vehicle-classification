"""
Inference utilities for the zero-shot vehicle benchmark - Fixed to let classifier do its job
"""
import time
from typing import Dict, List, Tuple, Any, Callable

import numpy as np

from models.classifiers.classifier_factory import ClassifierFactory
from models.detectors.detector_factory import DetectorFactory
from models.end_to_end.end_to_end_factory import EndToEndFactory

from utils.image_preprocessing import enhance_image

class InferencePipeline:
    """
    Inference pipeline for running detector+classifier combinations
    or end-to-end models for vehicle detection and classification
    """
    
    def __init__(self, detector_name: str = None, classifier_name: str = None, end_to_end: bool = False, **kwargs):
        """
        Initialize inference pipeline
        
        Args:
            detector_name: Name of the detector to use
            classifier_name: Name of the classifier to use
            end_to_end: Whether to use an end-to-end model
            **kwargs: Additional arguments to pass to the detector/classifier constructors
        """
        self.end_to_end = end_to_end
        
        detector_kwargs = {k: v for k, v in kwargs.items() if k in ['model_size', 'confidence_threshold', 'custom_model_path']}
        classifier_kwargs = {k: v for k, v in kwargs.items() if k in ['model_size']}
        
        if end_to_end:
            # Use EndToEndFactory to create end-to-end model
            self.model = EndToEndFactory.create_detector(
                detector_type=detector_name,
                **detector_kwargs
            )
                
            self.detector_name = detector_name
            self.classifier_name = None
        else:
            # Use DetectorFactory to create detector
            self.detector = DetectorFactory.create_detector(
                detector_type=detector_name,
                **detector_kwargs
            )
                
            # Use ClassifierFactory to create classifier
            self.classifier = ClassifierFactory.create_classifier(
                classifier_type=classifier_name,
                **classifier_kwargs
            )
                
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
        
        # REMOVED: image = enhance_image(image)
        
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
            # Run detector - ONLY FOR BOUNDING BOXES, NO CLASS OVERRIDE
            detections, detection_time = self.detector.detect(image)
            
            # Find main vehicle based on bounding box position and size only
            main_vehicle = self.detector.find_main_vehicle(detections, image.shape)
            
            if main_vehicle is None:
                # No objects detected
                classification_time = 0.0
                main_vehicle_class = None
                class_confidence = None
                all_scores = {}
            else:
                # Crop main vehicle for classification
                cropped_vehicle = self.detector.crop_bbox(image, main_vehicle['bbox'])
                
                # Let the classifier do its job - NO INTERFERENCE
                classification, classification_time = self.classifier.classify(cropped_vehicle)
                
                # Extract classification results - trust the classifier completely
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

def create_pipeline(detector: str, classifier: str = None, **kwargs) -> InferencePipeline:
    """
    Factory function to create a pipeline
    
    Args:
        detector: Name of the detector or end-to-end model
        classifier: Name of the classifier (None for end-to-end models)
        **kwargs: Additional arguments to pass to the detector/classifier constructors
        
    Returns:
        InferencePipeline instance
    """
    # Determine if this is an end-to-end model by checking if it's in the EndToEndFactory's list
    end_to_end_models = EndToEndFactory.get_available_detector_types()
    end_to_end = classifier is None and detector.lower() in [model.lower() for model in end_to_end_models]
    
    return InferencePipeline(detector, classifier, end_to_end, **kwargs)