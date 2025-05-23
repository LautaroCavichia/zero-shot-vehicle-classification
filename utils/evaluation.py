"""
Evaluation utilities for the zero-shot vehicle benchmark.

This module provides comprehensive evaluation metrics and ground truth validation
for vehicle detection and classification performance assessment.
"""
import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, 
    confusion_matrix, classification_report
)

from config import VEHICLE_CLASSES

logger = logging.getLogger(__name__)


class GroundTruthValidator:
    """Validates ground truth annotations for evaluation."""
    
    @staticmethod
    def validate_ground_truth(ground_truth: Dict[str, Any]) -> bool:
        """
        Validate ground truth annotation structure.
        
        Args:
            ground_truth: Ground truth annotation dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            required_keys = ['image_id', 'annotations', 'main_vehicle']
            if not all(key in ground_truth for key in required_keys):
                logger.warning(f"Missing required keys in ground truth: {required_keys}")
                return False
            
            # Validate main vehicle if present
            main_vehicle = ground_truth.get('main_vehicle')
            if main_vehicle is not None:
                required_vehicle_keys = ['category', 'bbox', 'area']
                if not all(key in main_vehicle for key in required_vehicle_keys):
                    logger.warning(f"Invalid main vehicle structure: {main_vehicle}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ground truth validation failed: {e}")
            return False


class PredictionValidator:
    """Validates prediction results for evaluation."""
    
    @staticmethod
    def validate_prediction(prediction: Dict[str, Any]) -> bool:
        """
        Validate prediction structure.
        
        Args:
            prediction: Prediction dictionary from pipeline
            
        Returns:
            True if valid, False otherwise
        """
        try:
            required_keys = ['pipeline', 'main_vehicle_class', 'main_vehicle_confidence']
            if not all(key in prediction for key in required_keys):
                logger.warning(f"Missing required keys in prediction: {required_keys}")
                return False
            
            # Validate confidence score
            confidence = prediction.get('main_vehicle_confidence')
            if confidence is not None and (confidence < 0.0 or confidence > 1.0):
                logger.warning(f"Invalid confidence score: {confidence}")
                return False
            
            # Validate class name
            predicted_class = prediction.get('main_vehicle_class')
            if predicted_class is not None and predicted_class not in VEHICLE_CLASSES:
                logger.warning(f"Invalid predicted class: {predicted_class}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Prediction validation failed: {e}")
            return False


class MetricsCalculator:
    """Calculates various evaluation metrics."""
    
    @staticmethod
    def calculate_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary of calculated metrics
        """
        try:
            # Handle empty predictions
            if not y_true or not y_pred:
                logger.warning("Empty predictions or ground truth provided")
                return {
                    'accuracy': 0.0,
                    'macro_f1': 0.0,
                    'weighted_f1': 0.0,
                    'macro_precision': 0.0,
                    'macro_recall': 0.0,
                    'weighted_precision': 0.0,
                    'weighted_recall': 0.0
                }
            
            # Get unique labels present in the data
            labels = sorted(set(y_true) | set(y_pred))
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels),
                'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels),
                'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels),
                'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels),
                'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels),
                'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate classification metrics: {e}")
            return {}
    
    @staticmethod
    def calculate_per_class_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class precision, recall, and F1 scores.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary of per-class metrics
        """
        try:
            if not y_true or not y_pred:
                return {}
            
            labels = sorted(set(y_true) | set(y_pred))
            class_metrics = {}
            
            for cls in labels:
                # Create binary classification problem for this class
                cls_y_true = [1 if y == cls else 0 for y in y_true]
                cls_y_pred = [1 if y == cls else 0 for y in y_pred]
                
                class_metrics[cls] = {
                    'precision': precision_score(cls_y_true, cls_y_pred, zero_division=0),
                    'recall': recall_score(cls_y_true, cls_y_pred, zero_division=0),
                    'f1': f1_score(cls_y_true, cls_y_pred, zero_division=0),
                    'support': sum(cls_y_true)  # Number of true instances
                }
            
            return class_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate per-class metrics: {e}")
            return {}
    
    @staticmethod
    def calculate_confusion_matrix(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """
        Calculate confusion matrix with proper label handling.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary containing confusion matrix and labels
        """
        try:
            if not y_true or not y_pred:
                return {}
            
            labels = sorted(set(y_true) | set(y_pred))
            
            # Skip confusion matrix if only one class present
            if len(labels) <= 1:
                logger.info(f"Skipping confusion matrix - only one class detected: {labels}")
                return {}
            
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            return {
                'matrix': cm.tolist(),
                'labels': labels,
                'normalized_matrix': (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate confusion matrix: {e}")
            return {}


class BenchmarkEvaluator:
    """
    Comprehensive evaluator for benchmarking vehicle detection and classification performance.
    
    This class handles evaluation of predictions against ground truth with proper validation,
    metric calculation, and error handling.
    """
    
    def __init__(self):
        """Initialize the benchmark evaluator."""
        self.results = defaultdict(list)
        self.pipeline_results = defaultdict(lambda: defaultdict(list))
        
        # Initialize validators and calculator
        self.gt_validator = GroundTruthValidator()
        self.pred_validator = PredictionValidator()
        self.metrics_calculator = MetricsCalculator()
        
        logger.info("Initialized BenchmarkEvaluator")
    
    def evaluate_prediction(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single prediction against ground truth.
        
        Args:
            prediction: Prediction dictionary from pipeline
            ground_truth: Ground truth annotation dictionary
            
        Returns:
            Evaluation result dictionary
        """
        try:
            # Validate inputs
            if not self.pred_validator.validate_prediction(prediction):
                logger.warning("Invalid prediction format, skipping evaluation")
                return {}
            
            if not self.gt_validator.validate_ground_truth(ground_truth):
                logger.warning("Invalid ground truth format, skipping evaluation")
                return {}
            
            # Extract pipeline name
            pipeline_name = prediction.get('pipeline', 'unknown')
            
            # Extract predictions
            pred_class = prediction.get('main_vehicle_class')
            pred_confidence = prediction.get('main_vehicle_confidence', 0.0)
            
            # Extract ground truth
            main_vehicle_gt = ground_truth.get('main_vehicle')
            if main_vehicle_gt is not None:
                gt_class = main_vehicle_gt.get('category', 'non-vehicle')
            else:
                gt_class = 'non-vehicle'
            
            # Ensure lowercase consistency
            if pred_class:
                pred_class = pred_class.lower()
            gt_class = gt_class.lower()
            
            # Calculate correctness
            correct = pred_class == gt_class
            
            # Extract timing information
            detection_time = prediction.get('detection_time', 0.0)
            classification_time = prediction.get('classification_time', 0.0)
            total_time = prediction.get('total_time', 0.0)
            
            # Create evaluation result
            result = {
                'pipeline': pipeline_name,
                'predicted_class': pred_class,
                'ground_truth_class': gt_class,
                'correct': correct,
                'confidence': pred_confidence,
                'detection_time': detection_time,
                'classification_time': classification_time,
                'total_time': total_time,
                'image_id': ground_truth.get('image_id', -1)
            }
            
            # Store results
            self.results['all'].append(result)
            self.pipeline_results[pipeline_name]['results'].append(result)
            
            logger.debug(f"Evaluated prediction: {pipeline_name} - "
                        f"Predicted: {pred_class}, GT: {gt_class}, Correct: {correct}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate prediction: {e}")
            return {}
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive metrics across all evaluated predictions.
        
        Returns:
            Dictionary containing metrics for all pipelines and overall results
        """
        try:
            metrics = {}
            
            # Compute overall metrics
            all_results = self.results.get('all', [])
            if all_results:
                metrics['overall'] = self._compute_pipeline_metrics(all_results)
                logger.info(f"Computed overall metrics from {len(all_results)} predictions")
            
            # Compute metrics per pipeline
            for pipeline_name, pipeline_data in self.pipeline_results.items():
                pipeline_results = pipeline_data.get('results', [])
                if pipeline_results:
                    metrics[pipeline_name] = self._compute_pipeline_metrics(pipeline_results)
                    logger.info(f"Computed metrics for {pipeline_name}: {len(pipeline_results)} predictions")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to compute metrics: {e}")
            return {}
    
    def _compute_pipeline_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for a single pipeline.
        
        Args:
            results: List of evaluation result dictionaries
            
        Returns:
            Dictionary containing computed metrics
        """
        try:
            if not results:
                logger.warning("No results provided for metric computation")
                return {}
            
            # Extract data for metric calculation
            y_true = [r['ground_truth_class'] for r in results]
            y_pred = [r['predicted_class'] or 'non-vehicle' for r in results]
            confidences = [r['confidence'] or 0.0 for r in results]
            detection_times = [r['detection_time'] for r in results]
            classification_times = [r['classification_time'] for r in results]
            total_times = [r['total_time'] for r in results]
            
            # Calculate classification metrics
            classification_metrics = self.metrics_calculator.calculate_classification_metrics(y_true, y_pred)
            
            # Calculate per-class metrics
            per_class_metrics = self.metrics_calculator.calculate_per_class_metrics(y_true, y_pred)
            
            # Calculate confusion matrix
            confusion_matrix_data = self.metrics_calculator.calculate_confusion_matrix(y_true, y_pred)
            
            # Calculate timing and confidence statistics
            metrics = {
                **classification_metrics,
                'avg_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
                'avg_detection_time': float(np.mean(detection_times)),
                'std_detection_time': float(np.std(detection_times)),
                'avg_classification_time': float(np.mean(classification_times)),
                'std_classification_time': float(np.std(classification_times)),
                'avg_total_time': float(np.mean(total_times)),
                'std_total_time': float(np.std(total_times)),
                'count': len(results),
                'class_metrics': per_class_metrics
            }
            
            # Add confusion matrix if available
            if confusion_matrix_data:
                metrics['confusion_matrix'] = confusion_matrix_data
            
            # Calculate accuracy per class for additional insights
            class_accuracy = {}
            for cls in set(y_true):
                cls_indices = [i for i, true_cls in enumerate(y_true) if true_cls == cls]
                if cls_indices:
                    cls_correct = sum(1 for i in cls_indices if y_true[i] == y_pred[i])
                    class_accuracy[cls] = cls_correct / len(cls_indices)
            
            metrics['class_accuracy'] = class_accuracy
            
            logger.debug(f"Computed metrics for {len(results)} results: "
                        f"Accuracy={metrics.get('accuracy', 0):.3f}, "
                        f"F1={metrics.get('macro_f1', 0):.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to compute pipeline metrics: {e}")
            return {}
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the evaluation results.
        
        Returns:
            Summary dictionary with key statistics
        """
        try:
            summary = {
                'total_evaluations': len(self.results.get('all', [])),
                'pipelines_evaluated': len(self.pipeline_results),
                'pipeline_names': list(self.pipeline_results.keys())
            }
            
            # Add per-pipeline summary
            pipeline_summaries = {}
            for pipeline_name, pipeline_data in self.pipeline_results.items():
                results = pipeline_data.get('results', [])
                if results:
                    pipeline_summaries[pipeline_name] = {
                        'total_predictions': len(results),
                        'correct_predictions': sum(1 for r in results if r.get('correct', False)),
                        'accuracy': sum(1 for r in results if r.get('correct', False)) / len(results)
                    }
            
            summary['pipeline_summaries'] = pipeline_summaries
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate evaluation summary: {e}")
            return {}
    
    def reset(self) -> None:
        """Reset the evaluator state."""
        self.results.clear()
        self.pipeline_results.clear()
        logger.info("Evaluator state reset")


def create_evaluator() -> BenchmarkEvaluator:
    """
    Factory function to create a benchmark evaluator.
    
    Returns:
        BenchmarkEvaluator instance
    """
    return BenchmarkEvaluator()