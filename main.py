"""
Main script for the zero-shot vehicle detection and classification benchmark
"""
import os
import argparse
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import gc
import torch

import numpy as np
from tqdm import tqdm
from scipy import stats

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
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of times to run the benchmark (default: 1)')
    
    return parser.parse_args()

def clear_memory():
    """Clear cache and force garbage collection between pipeline runs"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # For Apple Silicon / Metal
    elif torch.backends.mps.is_available():
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()


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
        ('yolov12', 'siglip'),

        ('supervision', 'clip'),
        ('supervision', 'openclip'),
        ('supervision', 'git'),
        ('supervision', 'siglip'),
   
        ('ssd', 'clip'),
        ('ssd', 'openclip'),
        ('ssd', 'git'),
        ('ssd', 'siglip'),

        
        # End-to-end models
        # ('dino', None),
        # ('owlv2', None),
        ('yolo_world', None),
    ]
    
     # Create evaluator for overall results
    overall_evaluator = BenchmarkEvaluator()
    
    # Limit number of images if specified
    num_images = min(len(dataset), args.limit) if args.limit else len(dataset)
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Store results for all runs
    all_runs_metrics = []
    
    # Progress tracking
    start_time = datetime.now()
    total_iterations = args.num_runs * len(pipelines) * num_images
    completed_iterations = 0
    
    # Run benchmark for specified number of runs
    for run in range(args.num_runs):
        print(f"\n{'='*50}")
        print(f"Run {run + 1}/{args.num_runs}")
        print(f"{'='*50}")
        
        # Create evaluator for this run
        run_evaluator = BenchmarkEvaluator()
        
        # Run benchmark for each pipeline
        for pipeline_idx, (detector_name, classifier_name) in enumerate(pipelines):
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
                    
                    # Evaluate prediction for both evaluators
                    result = run_evaluator.evaluate_prediction(prediction, annotations)
                    overall_evaluator.evaluate_prediction(prediction, annotations)
                    
                    # Save visualization if requested (only on first run to avoid duplicates)
                    if args.save_visualizations and run == 0:
                        vis_dir = Path(args.results_dir) / 'visualizations' / pipeline_name
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        image_name = Path(image_path).name
                        vis_path = vis_dir / f"{Path(image_name).stem}_pred.png"
                        
                        visualize_detection(image, prediction, str(vis_path))
                    
                    # Update progress
                    completed_iterations += 1
                    if completed_iterations % 100 == 0:  # Update every 50 iterations to reduce overhead
                        elapsed_time = datetime.now() - start_time
                        iterations_per_second = completed_iterations / elapsed_time.total_seconds()
                        remaining_iterations = total_iterations - completed_iterations
                        
                        if iterations_per_second > 0:
                            estimated_remaining = timedelta(seconds=remaining_iterations / iterations_per_second)
                            print(f"\nOverall Progress: {completed_iterations}/{total_iterations} "
                                  f"({completed_iterations/total_iterations*100:.1f}%), "
                                  f"ETA: {estimated_remaining}")
                    
            except Exception as e:  
                print(f"Error running pipeline {pipeline_name}: {e}")
                continue
            
            finally:
                clear_memory()
        
        # Compute metrics for this run
        run_metrics = run_evaluator.compute_metrics()
        all_runs_metrics.append(run_metrics)
        
        # Save individual run results
        run_results_file = Path(args.results_dir) / f'benchmark_results_run_{run + 1}.json'
        with open(run_results_file, 'w') as f:
            json.dump(run_metrics, f, indent=2)
    
    # Compute overall metrics (averaged across all runs)
    print("\nComputing overall metrics...")
    overall_metrics = overall_evaluator.compute_metrics()
    
    # Save overall metrics
    results_file = Path(args.results_dir) / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    # Save averaged metrics if multiple runs
    if args.num_runs > 1:
        averaged_metrics = average_metrics(all_runs_metrics)
        averaged_results_file = Path(args.results_dir) / 'benchmark_results_averaged.json'
        with open(averaged_results_file, 'w') as f:
            json.dump(averaged_metrics, f, indent=2)
        print(f"Averaged results saved to {averaged_results_file}")
        
        # Perform statistical analysis
        statistical_results = analyze_statistical_significance(all_runs_metrics)
        stats_file = Path(args.results_dir) / 'statistical_analysis.json'
        with open(stats_file, 'w') as f:
            json.dump(statistical_results, f, indent=2)
        print(f"Statistical analysis saved to {stats_file}")
        
        # Create human-readable summary
        create_statistical_summary(statistical_results, args.results_dir)
    
    print(f"Overall results saved to {results_file}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(overall_metrics, args.results_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK COMPLETED")
    print(f"Total time: {datetime.now() - start_time}")
    print(f"Total iterations: {completed_iterations}")
    print("="*50)
    
    
def analyze_statistical_significance(all_runs_metrics):
    """Perform statistical analysis between pipelines"""
    results = {}
    
    # Get pipeline names (excluding 'all' if it exists)
    pipelines = list(all_runs_metrics[0].keys())
    pipelines = [p for p in pipelines if p != 'all']
    
    # Collect accuracy data for each pipeline
    pipeline_accuracies = {}
    for pipeline in pipelines:
        accuracies = []
        for run in all_runs_metrics:
            if pipeline in run and 'accuracy' in run[pipeline]:
                accuracies.append(run[pipeline]['accuracy'])
        pipeline_accuracies[pipeline] = accuracies
    
    # Perform pairwise t-tests
    for i, pipeline1 in enumerate(pipelines):
        for pipeline2 in pipelines[i+1:]:
            acc1 = pipeline_accuracies[pipeline1]
            acc2 = pipeline_accuracies[pipeline2]
            
            if len(acc1) > 1 and len(acc2) > 1:
                t_stat, p_value = stats.ttest_ind(acc1, acc2)
                
                comparison_key = f"{pipeline1}_vs_{pipeline2}"
                results[comparison_key] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05),
                    'mean_diff': float(np.mean(acc1) - np.mean(acc2)),
                    'pipeline1_mean': float(np.mean(acc1)),
                    'pipeline1_std': float(np.std(acc1)),
                    'pipeline2_mean': float(np.mean(acc2)),
                    'pipeline2_std': float(np.std(acc2))
                }
    
    # Add ANOVA test for all pipelines
    if len(pipelines) > 2 and all(len(pipeline_accuracies[p]) > 1 for p in pipelines):
        accuracy_arrays = [pipeline_accuracies[p] for p in pipelines]
        f_stat, p_value = stats.f_oneway(*accuracy_arrays)
        
        results['anova'] = {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'pipelines': pipelines
        }
    
    # Find best performing pipeline
    mean_accuracies = {p: np.mean(accs) for p, accs in pipeline_accuracies.items() if len(accs) > 0}
    if mean_accuracies:
        best_pipeline = max(mean_accuracies.items(), key=lambda x: x[1])
        results['best_pipeline'] = {
            'name': best_pipeline[0],
            'mean_accuracy': float(best_pipeline[1]),
            'std_accuracy': float(np.std(pipeline_accuracies[best_pipeline[0]]))
        }
    
    return results
    
    
def average_metrics(all_runs_metrics):
    """Average metrics across multiple runs"""
    if not all_runs_metrics:
        return {}
    
    averaged = {}
    
    # Get all pipelines present in all runs
    all_pipelines = set()
    for run_metrics in all_runs_metrics:
        all_pipelines.update(run_metrics.keys())
    
    # Average metrics for each pipeline
    for pipeline in all_pipelines:
        pipeline_metrics = []
        for run_metrics in all_runs_metrics:
            if pipeline in run_metrics:
                pipeline_metrics.append(run_metrics[pipeline])
        
        if not pipeline_metrics:
            continue
        
        # Average numeric metrics
        averaged[pipeline] = {}
        numeric_keys = ['accuracy', 'macro_f1', 'weighted_f1', 'avg_confidence', 
                       'avg_detection_time', 'avg_classification_time', 'avg_total_time']
        
        for key in numeric_keys:
            values = [m[key] for m in pipeline_metrics if key in m]
            if values:
                averaged[pipeline][key] = float(np.mean(values))
                averaged[pipeline][f'{key}_std'] = float(np.std(values))
                averaged[pipeline][f'{key}_min'] = float(np.min(values))
                averaged[pipeline][f'{key}_max'] = float(np.max(values))
                averaged[pipeline][f'{key}_median'] = float(np.median(values))
        
        # Handle count
        if 'count' in pipeline_metrics[0]:
            averaged[pipeline]['count'] = pipeline_metrics[0]['count']
        
        # Average class metrics
        if 'class_metrics' in pipeline_metrics[0]:
            class_metrics = {}
            all_classes = set()
            for m in pipeline_metrics:
                if 'class_metrics' in m:
                    all_classes.update(m['class_metrics'].keys())
            
            for cls in all_classes:
                class_metrics[cls] = {}
                for metric in ['precision', 'recall', 'f1']:
                    values = []
                    for m in pipeline_metrics:
                        if 'class_metrics' in m and cls in m['class_metrics']:
                            values.append(m['class_metrics'][cls][metric])
                    if values:
                        class_metrics[cls][metric] = float(np.mean(values))
                        class_metrics[cls][f'{metric}_std'] = float(np.std(values))
            
            averaged[pipeline]['class_metrics'] = class_metrics
        
        # Add run information
        averaged[pipeline]['num_runs'] = len(pipeline_metrics)
    
    return averaged

def create_statistical_summary(statistical_results, output_dir):
    """Create a human-readable statistical summary"""
    summary_file = Path(output_dir) / 'statistical_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("Statistical Analysis Summary\n")
        f.write("==========================\n\n")
        
        # Best pipeline
        if 'best_pipeline' in statistical_results:
            best = statistical_results['best_pipeline']
            f.write(f"Best Performing Pipeline: {best['name']}\n")
            f.write(f"Mean Accuracy: {best['mean_accuracy']:.4f} ± {best['std_accuracy']:.4f}\n\n")
        
        # ANOVA results
        if 'anova' in statistical_results:
            anova = statistical_results['anova']
            f.write("ANOVA Test Results:\n")
            f.write(f"F-statistic: {anova['f_statistic']:.4f}\n")
            f.write(f"p-value: {anova['p_value']:.6f}\n")
            f.write(f"Significant differences between pipelines: {'Yes' if anova['significant'] else 'No'}\n\n")
        
        # Pairwise comparisons
        f.write("Pairwise Comparisons (t-tests):\n")
        f.write("-" * 40 + "\n")
        
        for comparison, data in statistical_results.items():
            if comparison not in ['anova', 'best_pipeline']:
                pipelines = comparison.split('_vs_')
                f.write(f"\n{pipelines[0]} vs {pipelines[1]}:\n")
                f.write(f"  Mean accuracy difference: {data['mean_diff']:.4f}\n")
                f.write(f"  t-statistic: {data['t_statistic']:.4f}\n")
                f.write(f"  p-value: {data['p_value']:.6f}\n")
                f.write(f"  Significantly different: {'Yes' if data['significant'] else 'No'}\n")
                f.write(f"  {pipelines[0]}: {data['pipeline1_mean']:.4f} ± {data['pipeline1_std']:.4f}\n")
                f.write(f"  {pipelines[1]}: {data['pipeline2_mean']:.4f} ± {data['pipeline2_std']:.4f}\n")
    
    print(f"Statistical summary saved to {summary_file}")

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