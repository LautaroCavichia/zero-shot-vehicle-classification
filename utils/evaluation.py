"""
Evaluation utilities for the zero-shot vehicle benchmark
"""
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


class BenchmarkEvaluator:
    """Evaluator for benchmarking performance of vehicle detection and classification"""
    
    def __init__(self):
        """Initialize the evaluator"""
        # Initialize metrics
        self.results = defaultdict(list)
        self.pipeline_results = defaultdict(lambda: defaultdict(list))
    
    def evaluate_prediction(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single prediction against ground truth
        
        Args:
            prediction: Prediction dict from the pipeline
            ground_truth: Ground truth annotation
            
        Returns:
            Dict with evaluation metrics
        """
        pipeline_name = prediction['pipeline']
        
        # Extract main vehicle predictions
        pred_class = prediction.get('main_vehicle_class')
        pred_confidence = prediction.get('main_vehicle_confidence', 0.0)
        
        # Extract ground truth
        if ground_truth.get('main_vehicle') is not None:
            gt_class = ground_truth['main_vehicle'].get('category', 'non-vehicle')
        else:
            gt_class = 'non-vehicle'
        
        # Calculate metrics
        correct = pred_class == gt_class
        
        # Store result
        result = {
            'pipeline': pipeline_name,
            'predicted_class': pred_class,
            'ground_truth_class': gt_class,
            'correct': correct,
            'confidence': pred_confidence,
            'detection_time': prediction.get('detection_time', 0.0),
            'classification_time': prediction.get('classification_time', 0.0),
            'total_time': prediction.get('total_time', 0.0),
        }
        
        # Add to results
        self.results['all'].append(result)
        self.pipeline_results[pipeline_name]['results'].append(result)
        
        return result
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics across all evaluated predictions
        
        Returns:
            Dict with aggregated metrics
        """
        metrics = {}
        
        # Compute metrics for all results
        all_results = self.results['all']
        metrics['all'] = self._compute_pipeline_metrics(all_results)
        
        # Compute metrics per pipeline
        for pipeline_name, pipeline_data in self.pipeline_results.items():
            pipeline_results = pipeline_data['results']
            metrics[pipeline_name] = self._compute_pipeline_metrics(pipeline_results)
        
        return metrics
        
    def _compute_pipeline_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute metrics for a single pipeline
        
        Args:
            results: List of result dicts
            
        Returns:
            Dict with metrics
        """
        if not results:
            return {}
        
        # Extract data
        y_true = [r['ground_truth_class'] for r in results]
        y_pred = [r['predicted_class'] or 'non-vehicle' for r in results]
        confidences = [r['confidence'] or 0.0 for r in results]
        detection_times = [r['detection_time'] for r in results]
        classification_times = [r['classification_time'] for r in results]
        total_times = [r['total_time'] for r in results]
        
        # Find all unique classes for proper label passing
        all_classes = sorted(set(y_true) | set(y_pred))
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0, labels=all_classes),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_classes),
            'avg_confidence': np.mean(confidences),
            'avg_detection_time': np.mean(detection_times),
            'avg_classification_time': np.mean(classification_times),
            'avg_total_time': np.mean(total_times),
            'count': len(results),
        }
        
        # Try to compute per-class metrics
        try:
            class_metrics = {}
            
            for cls in all_classes:
                cls_y_true = [1 if y == cls else 0 for y in y_true]
                cls_y_pred = [1 if y == cls else 0 for y in y_pred]
                
                class_metrics[cls] = {
                    'precision': precision_score(cls_y_true, cls_y_pred, zero_division=0),
                    'recall': recall_score(cls_y_true, cls_y_pred, zero_division=0),
                    'f1': f1_score(cls_y_true, cls_y_pred, zero_division=0),
                }
            
            metrics['class_metrics'] = class_metrics
            
            # Compute confusion matrix, explicitly passing all labels
            # Skip if only one class is present
            if len(all_classes) > 1:
                cm = confusion_matrix(y_true, y_pred, labels=all_classes)
                metrics['confusion_matrix'] = {
                    'matrix': cm.tolist(),
                    'labels': all_classes,
                }
            else:
                print(f"Skipping confusion matrix - only one class detected: {all_classes}")
        except Exception as e:
            print(f"Error computing per-class metrics: {e}")
        
        return metrics