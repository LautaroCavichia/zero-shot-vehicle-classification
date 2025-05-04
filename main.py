"""
Main script for the zero-shot vehicle detection and classification benchmark
"""
import os
import argparse
import time
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import RESULTS_DIR, VEHICLE_CLASSES
from utils.data_loader import create_dataloader
from utils.inference import create_pipeline
from utils.evaluation import BenchmarkEvaluator
from utils.visualization import visualize_results, visualize_detection


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Zero-shot vehicle detection and classification benchmark')
    
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--results_dir', type=str, default=str(RESULTS_DIR),
                        help='Path to results directory')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of images to process')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save detection visualizations')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Only generate visualizations from existing results')
    
    return parser.parse_args()


def run_benchmark(args):
    """Run the benchmark with all pipelines"""
    # Create data loader
    print("Loading dataset...")
    dataset = create_dataloader()
    
    # Define pipelines to benchmark
    pipelines = [
        # Detector + Classifier combinations
        ('yolov12', 'clip'),
        ('yolov12', 'openclip'),
        ('yolov12', 'git'),
        # ('yolov12', 'vilt'),
        ('supervision', 'clip'),
        ('supervision', 'openclip'),
        ('supervision', 'git'),
        # ('supervision', 'vilt'),
        ('ssd', 'clip'),
        ('ssd', 'openclip'),
        ('ssd', 'git'),
        # ('ssd', 'vilt'),
        
        # End-to-end models
        # ('glip', None),
        ('yolo_world', None),
    ]
    
    # Create evaluator
    evaluator = BenchmarkEvaluator()
    
    # Limit number of images if specified
    num_images = min(len(dataset), args.limit) if args.limit else len(dataset)
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Path to store results
    results_file = Path(args.results_dir) / 'benchmark_results.json'
    
    # Run benchmark for each pipeline
    for detector_name, classifier_name in pipelines:
        pipeline_name = f"{detector_name}+{classifier_name}" if classifier_name else detector_name
        print(f"\nRunning benchmark with pipeline: {pipeline_name}")
        
        try:
            # Create pipeline
            pipeline = create_pipeline(detector_name, classifier_name)
            
            # Process each image
            for i in tqdm(range(num_images), desc=f"Processing images with {pipeline_name}"):
                # Load image
                image, image_path = dataset.load_image(i)
                
                # Get ground truth annotations
                annotations = dataset.get_annotations(i)
                
                # Run inference
                prediction = pipeline.process_image(image)
                
                # Evaluate prediction
                result = evaluator.evaluate_prediction(prediction, annotations)
                
                # Save visualization if requested
                if args.save_visualizations:
                    vis_dir = Path(args.results_dir) / 'visualizations' / pipeline_name
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    image_name = Path(image_path).name
                    vis_path = vis_dir / f"{Path(image_name).stem}_pred.png"
                    
                    visualize_detection(image, prediction, str(vis_path))
                
        except Exception as e:
            print(f"Error running pipeline {pipeline_name}: {e}")
            continue
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = evaluator.compute_metrics()
    
    # Save metrics
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(metrics, args.results_dir)


def visualize_from_results(args):
    """Generate visualizations from existing results"""
    # Path to stored results
    results_file = Path(args.results_dir) / 'benchmark_results.json'
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    # Load metrics
    with open(results_file, 'r') as f:
        metrics = json.load(f)
    
    # Visualize results
    print("Generating visualizations...")
    visualize_results(metrics, args.results_dir)


def main():
    """Main function"""
    args = parse_args()
    
    # Set environment variables for Apple Silicon compatibility
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    if args.visualize_only:
        visualize_from_results(args)
    else:
        run_benchmark(args)


if __name__ == '__main__':
    main()