"""
Streamlined visualization utilities for the zero-shot vehicle benchmark.

This module provides focused visualizations for accuracy vs speed analysis,
confusion matrices, and timing metrics without statistical complexity.
"""
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import pandas as pd

from config import RESULTS_DIR
from utils.image_preprocessing import VehicleImageEnhancer

logger = logging.getLogger(__name__)


class VisualizationConfig:
    """Configuration for consistent visualization styling."""
    
    # Color schemes
    PIPELINE_COLORS = sns.color_palette("Set1", 10)
    METRIC_COLORS = {
        'accuracy': '#2E86C1',
        'timing': '#E74C3C',
        'f1': '#28B463'
    }
    
    # Style settings
    FIGURE_DPI = 300
    FONT_SIZE = {
        'title': 16,
        'label': 14,
        'tick': 12,
        'legend': 11
    }
    
    # Plot dimensions
    SINGLE_PLOT_SIZE = (10, 8)
    COMPARISON_PLOT_SIZE = (12, 8)


class MetricsVisualizer:
    """Handles visualization of benchmark metrics."""
    
    def __init__(self, save_dir: str = None):
        """
        Initialize the metrics visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir) if save_dir else RESULTS_DIR
        self.save_dir.mkdir(exist_ok=True)
        self.config = VisualizationConfig()
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette(self.config.PIPELINE_COLORS)
        
        logger.info(f"Initialized MetricsVisualizer, saving to: {self.save_dir}")
    
    def create_accuracy_vs_speed_plot(self, metrics: Dict[str, Any]) -> None:
        """
        Create accuracy vs speed scatter plot.
        
        Args:
            metrics: Dictionary containing metrics for all pipelines
        """
        try:
            # Extract pipeline data (exclude 'overall' if present)
            pipeline_names = [name for name in metrics.keys() if name != 'overall']
            
            if not pipeline_names:
                logger.warning("No pipeline data found for accuracy vs speed plot")
                return
            
            # Create DataFrame for plotting
            plot_data = []
            for pipeline in pipeline_names:
                pipeline_metrics = metrics[pipeline]
                plot_data.append({
                    'pipeline': pipeline,
                    'accuracy': pipeline_metrics.get('accuracy', 0),
                    'avg_total_time': pipeline_metrics.get('avg_total_time', 0),
                    'weighted_f1': pipeline_metrics.get('weighted_f1', 0)
                })
            
            df = pd.DataFrame(plot_data)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=self.config.SINGLE_PLOT_SIZE)
            
            # Create scatter plot colored by F1 score
            scatter = ax.scatter(
                df['avg_total_time'],
                df['accuracy'],
                s=120,
                c=df['weighted_f1'],
                cmap='viridis',
                alpha=0.8,
                edgecolors='white',
                linewidth=2
            )
            
            # Add pipeline labels
            for i, row in df.iterrows():
                ax.annotate(
                    row['pipeline'],
                    (row['avg_total_time'], row['accuracy']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=self.config.FONT_SIZE['tick'],
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
            
            # Customize plot
            ax.set_xlabel('Average Total Time (seconds)', fontsize=self.config.FONT_SIZE['label'])
            ax.set_ylabel('Accuracy', fontsize=self.config.FONT_SIZE['label'])
            ax.set_title('Accuracy vs Speed Performance', fontsize=self.config.FONT_SIZE['title'], fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Weighted F1 Score', fontsize=self.config.FONT_SIZE['label'])
            cbar.ax.tick_params(labelsize=self.config.FONT_SIZE['tick'])
            
            # Set axis limits with padding
            x_margin = (df['avg_total_time'].max() - df['avg_total_time'].min()) * 0.1
            y_margin = (df['accuracy'].max() - df['accuracy'].min()) * 0.05
            
            ax.set_xlim(df['avg_total_time'].min() - x_margin, df['avg_total_time'].max() + x_margin)
            ax.set_ylim(max(0, df['accuracy'].min() - y_margin), min(1, df['accuracy'].max() + y_margin))
            
            plt.tight_layout()
            
            # Save plot
            save_path = self.save_dir / 'accuracy_vs_speed.png'
            plt.savefig(save_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved accuracy vs speed plot to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to create accuracy vs speed plot: {e}")
    
    def create_timing_comparison(self, metrics: Dict[str, Any]) -> None:
        """
        Create timing metrics comparison chart.
        
        Args:
            metrics: Dictionary containing metrics for all pipelines
        """
        try:
            # Extract pipeline data
            pipeline_names = [name for name in metrics.keys() if name != 'overall']
            
            if not pipeline_names:
                logger.warning("No pipeline data found for timing comparison")
                return
            
            # Prepare data
            timing_data = []
            for pipeline in pipeline_names:
                pipeline_metrics = metrics[pipeline]
                timing_data.append({
                    'pipeline': pipeline,
                    'detection_time': pipeline_metrics.get('avg_detection_time', 0),
                    'classification_time': pipeline_metrics.get('avg_classification_time', 0),
                    'total_time': pipeline_metrics.get('avg_total_time', 0)
                })
            
            df = pd.DataFrame(timing_data)
            
            # Create stacked bar chart
            fig, ax = plt.subplots(figsize=self.config.COMPARISON_PLOT_SIZE)
            
            # Create stacked bars
            bar_width = 0.6
            indices = np.arange(len(pipeline_names))
            
            p1 = ax.bar(indices, df['detection_time'], bar_width, 
                       label='Detection Time', color=self.config.METRIC_COLORS['timing'], alpha=0.8)
            p2 = ax.bar(indices, df['classification_time'], bar_width,
                       bottom=df['detection_time'], label='Classification Time', 
                       color=self.config.METRIC_COLORS['accuracy'], alpha=0.8)
            
            # Customize plot
            ax.set_xlabel('Pipeline', fontsize=self.config.FONT_SIZE['label'])
            ax.set_ylabel('Time (seconds)', fontsize=self.config.FONT_SIZE['label'])
            ax.set_title('Timing Performance Comparison', fontsize=self.config.FONT_SIZE['title'], fontweight='bold')
            ax.set_xticks(indices)
            ax.set_xticklabels(df['pipeline'], rotation=45, ha='right')
            ax.legend(fontsize=self.config.FONT_SIZE['legend'])
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (det_time, cls_time, total_time) in enumerate(zip(df['detection_time'], 
                                                                    df['classification_time'], 
                                                                    df['total_time'])):
                # Total time label on top
                ax.text(i, total_time + 0.001, f'{total_time:.3f}s', 
                       ha='center', va='bottom', fontsize=self.config.FONT_SIZE['tick'])
            
            plt.tight_layout()
            
            # Save plot
            save_path = self.save_dir / 'timing_comparison.png'
            plt.savefig(save_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved timing comparison to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to create timing comparison: {e}")
    
    def create_confusion_matrices(self, metrics: Dict[str, Any]) -> None:
        """
        Create confusion matrices for each pipeline.
        
        Args:
            metrics: Dictionary containing metrics for all pipelines
        """
        try:
            pipeline_names = [name for name in metrics.keys() if name != 'overall']
            
            for pipeline in pipeline_names:
                pipeline_metrics = metrics[pipeline]
                
                if 'confusion_matrix' not in pipeline_metrics:
                    logger.debug(f"No confusion matrix data for {pipeline}")
                    continue
                
                cm_data = pipeline_metrics['confusion_matrix']
                cm_matrix = np.array(cm_data['matrix'])
                labels = cm_data['labels']
                
                # Skip if matrix is too small
                if len(labels) <= 1:
                    logger.debug(f"Skipping confusion matrix for {pipeline} - insufficient classes")
                    continue
                
                # Create confusion matrix plot
                fig, ax = plt.subplots(figsize=self.config.SINGLE_PLOT_SIZE)
                
                # Normalize for better visualization
                cm_normalized = cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis]
                cm_normalized = np.nan_to_num(cm_normalized)
                
                # Create heatmap
                sns.heatmap(
                    cm_normalized,
                    annot=cm_matrix,  # Show actual counts
                    fmt='d',
                    cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels,
                    cbar_kws={'label': 'Normalized Frequency'},
                    ax=ax
                )
                
                # Customize plot
                ax.set_title(f'Confusion Matrix: {pipeline}', 
                           fontsize=self.config.FONT_SIZE['title'], fontweight='bold')
                ax.set_xlabel('Predicted Class', fontsize=self.config.FONT_SIZE['label'])
                ax.set_ylabel('True Class', fontsize=self.config.FONT_SIZE['label'])
                
                # Rotate labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                plt.setp(ax.get_yticklabels(), rotation=0)
                
                plt.tight_layout()
                
                # Save plot
                safe_pipeline_name = pipeline.replace('+', '_').replace('/', '_')
                save_path = self.save_dir / f'confusion_matrix_{safe_pipeline_name}.png'
                plt.savefig(save_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved confusion matrix for {pipeline} to {save_path}")
                
        except Exception as e:
            logger.error(f"Failed to create confusion matrices: {e}")
    
    def create_summary_table(self, metrics: Dict[str, Any]) -> None:
        """
        Create and save a summary table of key metrics.
        
        Args:
            metrics: Dictionary containing metrics for all pipelines
        """
        try:
            pipeline_names = [name for name in metrics.keys() if name != 'overall']
            
            if not pipeline_names:
                logger.warning("No pipeline data found for summary table")
                return
            
            # Prepare summary data
            summary_data = []
            for pipeline in pipeline_names:
                pipeline_metrics = metrics[pipeline]
                summary_data.append({
                    'Pipeline': pipeline,
                    'Accuracy': f"{pipeline_metrics.get('accuracy', 0):.3f}",
                    'Weighted F1': f"{pipeline_metrics.get('weighted_f1', 0):.3f}",
                    'Avg Total Time (s)': f"{pipeline_metrics.get('avg_total_time', 0):.3f}",
                    'Detection Time (s)': f"{pipeline_metrics.get('avg_detection_time', 0):.3f}",
                    'Classification Time (s)': f"{pipeline_metrics.get('avg_classification_time', 0):.3f}",
                    'Count': pipeline_metrics.get('count', 0)
                })
            
            # Create DataFrame and sort by accuracy
            df = pd.DataFrame(summary_data)
            df = df.sort_values('Accuracy', ascending=False)
            
            # Save as CSV
            csv_path = self.save_dir / 'benchmark_summary.csv'
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Saved summary table to {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to create summary table: {e}")


class DetectionVisualizer:
    """Handles visualization of individual detection results."""
    
    def __init__(self):
        """Initialize the detection visualizer."""
        self.config = VisualizationConfig()
        self.preprocessor = VehicleImageEnhancer()
        
    def visualize_detection(self, 
                          image: np.ndarray, 
                          prediction: Dict[str, Any], 
                          save_path: Optional[str] = None,
                          enhance_image: bool = True) -> None:
        """
        Visualize a single detection result.
        
        Args:
            image: Input image (RGB format)
            prediction: Prediction dictionary from pipeline
            save_path: Optional path to save visualization
            enhance_image: Whether to enhance image for better visualization
        """
        try:
            # Enhance image if requested
            if enhance_image:
                display_image = self.preprocessor.preprocess(image)
            else:
                display_image = image.copy()
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.config.SINGLE_PLOT_SIZE)
            ax.imshow(display_image)
            
            # Extract prediction information
            class_name = prediction.get('main_vehicle_class', 'Unknown')
            confidence = prediction.get('main_vehicle_confidence', 0.0)
            detection_time = prediction.get('detection_time', 0.0)
            classification_time = prediction.get('classification_time', 0.0)
            pipeline_name = prediction.get('pipeline', 'Unknown')
            
            # Set title
            title = (f"Pipeline: {pipeline_name}\n"
                    f"Class: {class_name} (Confidence: {confidence:.2f})\n"
                    f"Times - Detection: {detection_time:.3f}s, Classification: {classification_time:.3f}s")
            
            fig.suptitle(title, fontsize=self.config.FONT_SIZE['label'], y=0.95)
            
            # Draw bounding box if available
            main_vehicle = prediction.get('main_vehicle')
            if main_vehicle and 'bbox' in main_vehicle:
                bbox = main_vehicle['bbox']
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=3,
                        edgecolor='cyan',
                        facecolor='none',
                        alpha=0.8
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    label_text = f"{class_name}: {confidence:.2f}"
                    ax.text(
                        x1 + 5, y1 + 5,
                        label_text,
                        color='white',
                        fontsize=self.config.FONT_SIZE['tick'],
                        fontweight='bold',
                        bbox=dict(
                            boxstyle='round,pad=0.3',
                            facecolor='cyan',
                            alpha=0.8,
                            edgecolor='white'
                        )
                    )
            
            ax.axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.90])
            
            if save_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
                plt.close()
                logger.debug(f"Saved detection visualization to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to visualize detection: {e}")
            if save_path:
                plt.close()


def visualize_results(metrics: Dict[str, Any], save_dir: str = None) -> None:
    """
    Main function to generate all benchmark visualizations.
    
    Args:
        metrics: Dictionary containing benchmark metrics
        save_dir: Directory to save visualizations
    """
    try:
        logger.info("Generating benchmark visualizations...")
        
        # Initialize visualizer
        visualizer = MetricsVisualizer(save_dir)
        
        # Generate core visualizations
        visualizer.create_accuracy_vs_speed_plot(metrics)
        visualizer.create_timing_comparison(metrics)
        visualizer.create_confusion_matrices(metrics)
        visualizer.create_summary_table(metrics)
        
        logger.info(f"All visualizations saved to {visualizer.save_dir}")
        
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")


def visualize_detection(image: np.ndarray, 
                       prediction: Dict[str, Any], 
                       save_path: Optional[str] = None) -> None:
    """
    Visualize a single detection result.
    
    Args:
        image: Input image (RGB format)
        prediction: Prediction dictionary from pipeline
        save_path: Optional path to save visualization
    """
    visualizer = DetectionVisualizer()
    visualizer.visualize_detection(image, prediction, save_path)