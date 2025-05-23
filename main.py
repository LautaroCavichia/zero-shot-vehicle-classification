"""
Main script for the zero-shot vehicle detection and classification benchmark.

This script orchestrates the entire benchmarking process with proper error handling,
progress tracking, and streamlined result generation.
"""
import os
import argparse
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import gc

import torch
from tqdm import tqdm

from config import RESULTS_DIR, VEHICLE_CLASSES
from utils.data_loader import create_dataloader
from utils.inference import create_pipeline
from utils.evaluation import create_evaluator
from utils.visualization import visualize_results, visualize_detection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('benchmark.log')
    ]
)
logger = logging.getLogger(__name__)


class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    # Default pipeline configurations
    DETECTOR_CLASSIFIER_PIPELINES = [
        ('yolov12', 'clip'),
        ('yolov12', 'openclip'), 
        ('yolov12', 'siglip'),
        ('yolov12', 'git'),
        ('supervision', 'clip'),
        ('supervision', 'openclip'),
        ('supervision', 'siglip'), 
        ('supervision', 'git'),
        ('ssd', 'clip'),
        ('ssd', 'openclip'),
        ('ssd', 'siglip'),
        ('ssd', 'git'),
    ]
    
    END_TO_END_PIPELINES = [
        ('owlv2', None),        
        ('yolo_world', None),   
        # ('dino', None),         
    ]
    
    @classmethod
    def get_all_pipelines(cls) -> List[tuple]:
        """Get all configured pipelines."""
        return cls.DETECTOR_CLASSIFIER_PIPELINES + cls.END_TO_END_PIPELINES


class MemoryManager:
    """Handles memory management during benchmark execution."""
    
    @staticmethod
    def clear_cache():
        """Clear GPU and system caches."""
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()


class ProgressTracker:
    """Tracks and reports benchmark progress."""
    
    def __init__(self, total_iterations: int):
        """
        Initialize progress tracker.
        
        Args:
            total_iterations: Total number of iterations expected
        """
        self.total_iterations = total_iterations
        self.completed_iterations = 0
        self.start_time = datetime.now()
        self.last_update = self.start_time
        
    def update(self, increment: int = 1) -> None:
        """
        Update progress and log if necessary.
        
        Args:
            increment: Number of iterations to add
        """
        self.completed_iterations += increment
        current_time = datetime.now()
        
        # Log progress every 50 iterations or every minute
        if (self.completed_iterations % 50 == 0 or 
            (current_time - self.last_update).total_seconds() > 60):
            
            self._log_progress()
            self.last_update = current_time
    
    def _log_progress(self) -> None:
        """Log current progress with ETA."""
        elapsed_time = datetime.now() - self.start_time
        progress_percent = (self.completed_iterations / self.total_iterations) * 100
        
        if self.completed_iterations > 0:
            avg_time_per_iteration = elapsed_time.total_seconds() / self.completed_iterations
            remaining_iterations = self.total_iterations - self.completed_iterations
            eta = timedelta(seconds=remaining_iterations * avg_time_per_iteration)
            
            logger.info(f"Progress: {self.completed_iterations}/{self.total_iterations} "
                       f"({progress_percent:.1f}%) - ETA: {eta}")


class BenchmarkRunner:
    """Main benchmark execution class."""
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize benchmark runner.
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.config = BenchmarkConfig()
        self.memory_manager = MemoryManager()
        
        # Create results directory
        self.results_dir = Path(args.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized BenchmarkRunner with results dir: {self.results_dir}")
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Execute the complete benchmark.
        
        Returns:
            Dictionary containing all benchmark results
        """
        try:
            logger.info("Starting benchmark execution...")
            
            # Load dataset
            dataset = self._load_dataset()
            
            # Determine number of images to process
            num_images = min(len(dataset), self.args.limit) if self.args.limit else len(dataset)
            logger.info(f"Processing {num_images} images")
            
            # Get pipelines to test
            pipelines = self.config.get_all_pipelines()
            
            # Initialize progress tracking
            total_iterations = len(pipelines) * num_images
            progress_tracker = ProgressTracker(total_iterations)
            
            # Initialize evaluator
            evaluator = create_evaluator()
            
            # Run benchmark for each pipeline
            for pipeline_idx, (detector_name, classifier_name) in enumerate(pipelines):
                pipeline_name = f"{detector_name}+{classifier_name}" if classifier_name else detector_name
                
                logger.info(f"Running pipeline {pipeline_idx + 1}/{len(pipelines)}: {pipeline_name}")
                
                try:
                    self._run_pipeline(
                        detector_name, classifier_name, dataset, num_images, 
                        evaluator, progress_tracker
                    )
                except Exception as e:
                    logger.error(f"Pipeline {pipeline_name} failed: {e}")
                    continue
                finally:
                    self.memory_manager.clear_cache()
            
            # Compute final metrics
            logger.info("Computing final metrics...")
            metrics = evaluator.compute_metrics()
            
            # Save results
            self._save_results(metrics)
            
            # Generate visualizations
            logger.info("Generating visualizations...")
            visualize_results(metrics, str(self.results_dir))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            raise
    
    def _load_dataset(self) -> Any:
        """Load and validate the dataset."""
        try:
            logger.info("Loading dataset...")
            dataset = create_dataloader()
            
            # Log dataset statistics
            stats = dataset.get_dataset_statistics()
            logger.info(f"Dataset loaded successfully:")
            logger.info(f"  - Total images: {stats.get('total_images', 0)}")
            logger.info(f"  - Total annotations: {stats.get('total_annotations', 0)}")
            logger.info(f"  - Class distribution: {stats.get('class_distribution', {})}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _run_pipeline(self, 
                     detector_name: str, 
                     classifier_name: str, 
                     dataset: Any, 
                     num_images: int,
                     evaluator: Any,
                     progress_tracker: ProgressTracker) -> None:
        """
        Run a single pipeline on the dataset.
        
        Args:
            detector_name: Name of the detector
            classifier_name: Name of the classifier (None for end-to-end)
            dataset: Dataset instance
            num_images: Number of images to process
            evaluator: Benchmark evaluator
            progress_tracker: Progress tracking instance
        """
        pipeline_name = f"{detector_name}+{classifier_name}" if classifier_name else detector_name
        
        try:
            # Create pipeline
            pipeline = create_pipeline(detector_name, classifier_name)
            
            # Process images with progress bar
            with tqdm(range(num_images), desc=f"Processing {pipeline_name}") as pbar:
                for i in pbar:
                    try:
                        # Load image and annotations
                        image, image_path = dataset.load_image(i)
                        annotations = dataset.get_annotations(i)
                        
                        # Run inference
                        prediction = pipeline.process_image(image)
                        
                        # Evaluate prediction
                        evaluator.evaluate_prediction(prediction, annotations)
                        
                        # Save visualization if requested
                        if self.args.save_visualizations:
                            self._save_visualization(
                                image, prediction, image_path, pipeline_name, i
                            )
                        
                        # Update progress
                        progress_tracker.update()
                        
                    except Exception as e:
                        logger.warning(f"Failed to process image {i} with {pipeline_name}: {e}")
                        continue
            
            logger.info(f"Completed pipeline: {pipeline_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline {pipeline_name}: {e}")
            raise
    
    def _save_visualization(self, 
                          image: Any, 
                          prediction: Dict[str, Any], 
                          image_path: str, 
                          pipeline_name: str, 
                          image_idx: int) -> None:
        """
        Save visualization for a prediction.
        
        Args:
            image: Input image
            prediction: Prediction results
            image_path: Path to original image
            pipeline_name: Name of the pipeline
            image_idx: Image index
        """
        try:
            # Create visualization directory
            vis_dir = self.results_dir / 'visualizations' / pipeline_name
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate visualization filename
            image_name = Path(image_path).stem
            vis_path = vis_dir / f"{image_name}_{image_idx:04d}_pred.png"
            
            # Create visualization
            visualize_detection(image, prediction, str(vis_path))
            
        except Exception as e:
            logger.warning(f"Failed to save visualization for image {image_idx}: {e}")
    
    def _save_results(self, metrics: Dict[str, Any]) -> None:
        """
        Save benchmark results to file.
        
        Args:
            metrics: Computed metrics dictionary
        """
        try:
            # Save comprehensive results
            results_file = self.results_dir / 'benchmark_results.json'
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_file}")
            
            # Save human-readable summary
            self._save_summary(metrics)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _save_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Save a human-readable summary of results.
        
        Args:
            metrics: Computed metrics dictionary
        """
        try:
            summary_file = self.results_dir / 'benchmark_summary.txt'
            
            with open(summary_file, 'w') as f:
                f.write("Zero-Shot Vehicle Detection Benchmark Results\n")
                f.write("=" * 50 + "\n\n")
                
                # Pipeline results
                pipeline_names = [name for name in metrics.keys() if name != 'overall']
                
                if pipeline_names:
                    f.write("Pipeline Performance Summary:\n")
                    f.write("-" * 30 + "\n")
                    
                    # Sort by accuracy
                    sorted_pipelines = sorted(
                        pipeline_names,
                        key=lambda x: metrics[x].get('accuracy', 0),
                        reverse=True
                    )
                    
                    for pipeline in sorted_pipelines:
                        pipeline_metrics = metrics[pipeline]
                        f.write(f"\n{pipeline}:\n")
                        f.write(f"  Accuracy: {pipeline_metrics.get('accuracy', 0):.3f}\n")
                        f.write(f"  Weighted F1: {pipeline_metrics.get('weighted_f1', 0):.3f}\n")
                        f.write(f"  Avg Total Time: {pipeline_metrics.get('avg_total_time', 0):.3f}s\n")
                        f.write(f"  Count: {pipeline_metrics.get('count', 0)}\n")
                
                # Overall statistics
                if 'overall' in metrics:
                    overall = metrics['overall']
                    f.write(f"\n\nOverall Statistics:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Total Predictions: {overall.get('count', 0)}\n")
                    f.write(f"Overall Accuracy: {overall.get('accuracy', 0):.3f}\n")
                    f.write(f"Overall F1: {overall.get('weighted_f1', 0):.3f}\n")
                
                f.write(f"\n\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"Summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Zero-shot vehicle detection and classification benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir', type=str, default='data',
        help='Path to data directory containing images and annotations'
    )
    parser.add_argument(
        '--results_dir', type=str, default=str(RESULTS_DIR),
        help='Path to results directory for saving outputs'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit the number of images to process (useful for testing)'
    )
    parser.add_argument(
        '--save_visualizations', action='store_true',
        help='Save detection visualizations for each prediction'
    )
    parser.add_argument(
        '--visualize_only', action='store_true',
        help='Only generate visualizations from existing results'
    )
    parser.add_argument(
        '--log_level', type=str, default='ERROR',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set logging level'
    )
    
    return parser.parse_args()


def setup_environment(args: argparse.Namespace) -> None:
    """
    Set up the execution environment.
    
    Args:
        args: Parsed command line arguments
    """
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Set environment variables for compatibility
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Log system information
    logger.info(f"Python executable: {os.sys.executable}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Log device availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
    elif torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) available")
    else:
        logger.info("Using CPU only")


def visualize_from_existing_results(args: argparse.Namespace) -> None:
    """
    Generate visualizations from existing results.
    
    Args:
        args: Parsed command line arguments
    """
    results_file = Path(args.results_dir) / 'benchmark_results.json'
    
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return
    
    try:
        # Load existing results
        with open(results_file, 'r') as f:
            metrics = json.load(f)
        
        logger.info("Generating visualizations from existing results...")
        visualize_results(metrics, args.results_dir)
        logger.info("Visualization generation completed")
        
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")


def main() -> None:
    """Main entry point."""
    try:
        # Parse arguments and setup environment
        args = parse_arguments()
        setup_environment(args)
        
        logger.info("Starting Zero-Shot Vehicle Detection Benchmark")
        logger.info(f"Arguments: {vars(args)}")
        
        if args.visualize_only:
            # Only generate visualizations
            visualize_from_existing_results(args)
        else:
            # Run full benchmark
            runner = BenchmarkRunner(args)
            metrics = runner.run_benchmark()
            
            logger.info("Benchmark completed successfully!")
            
            # Print quick summary
            if 'overall' in metrics:
                overall = metrics['overall']
                logger.info(f"Overall Results - Accuracy: {overall.get('accuracy', 0):.3f}, "
                           f"F1: {overall.get('weighted_f1', 0):.3f}")
    
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == '__main__':
    main()  