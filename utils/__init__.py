"""
Utils package initialization for the zero-shot vehicle benchmark.

This module provides centralized access to all utility functions and classes
used throughout the benchmark system.
"""
import logging

# Configure logging for the utils package
logger = logging.getLogger(__name__)

# Import main utility functions and classes
from utils.data_loader import create_dataloader, COCOVehicleDataset
from utils.inference import create_pipeline, InferencePipeline
from utils.evaluation import create_evaluator, BenchmarkEvaluator
from utils.visualization import visualize_results, visualize_detection
from utils.image_preprocessing import VehicleImageEnhancer, enhance_image

# Define public API
__all__ = [
    # Factory functions
    'create_dataloader',
    'create_pipeline', 
    'create_evaluator',
    
    # Main classes
    'COCOVehicleDataset',
    'InferencePipeline',
    'BenchmarkEvaluator',
    'VehicleImageEnhancer',
    
    # Utility functions
    'enhance_image',
    'visualize_results',
    'visualize_detection',
]

# Log successful initialization
logger.debug("Utils package initialized successfully")

# Version information
__version__ = "1.0.0"