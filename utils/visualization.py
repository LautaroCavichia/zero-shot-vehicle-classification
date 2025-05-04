"""
Visualization utilities for the zero-shot vehicle benchmark
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Any
import pandas as pd

from config import RESULTS_DIR


def visualize_results(metrics: Dict[str, Any], save_dir: str = None):
    """
    Visualize benchmark results
    
    Args:
        metrics: Dict of benchmark metrics
        save_dir: Directory to save visualizations (defaults to RESULTS_DIR)
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract pipeline names (excluding 'all')
    pipeline_names = [name for name in metrics.keys() if name != 'all']
    
    # Create a DataFrame for easier plotting
    data = []
    for pipeline in pipeline_names:
        pipeline_metrics = metrics[pipeline]
        data.append({
            'pipeline': pipeline,
            'accuracy': pipeline_metrics.get('accuracy', 0),
            'macro_f1': pipeline_metrics.get('macro_f1', 0),
            'weighted_f1': pipeline_metrics.get('weighted_f1', 0),
            'avg_confidence': pipeline_metrics.get('avg_confidence', 0),
            'avg_detection_time': pipeline_metrics.get('avg_detection_time', 0),
            'avg_classification_time': pipeline_metrics.get('avg_classification_time', 0),
            'avg_total_time': pipeline_metrics.get('avg_total_time', 0),
        })
    
    df = pd.DataFrame(data)
    
    # Set color palette and style
    colors = sns.color_palette("viridis", len(pipeline_names))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
    
    # 1. Plot accuracy metrics
    plt.figure(figsize=(12, 8))
    
    accuracy_df = df.melt(
        id_vars=['pipeline'],
        value_vars=['accuracy', 'macro_f1', 'weighted_f1'],
        var_name='Metric',
        value_name='Value'
    )
    
    ax = sns.barplot(x='pipeline', y='Value', hue='Metric', data=accuracy_df, 
                palette=sns.color_palette("Blues_d", 3))
    plt.title('Accuracy Metrics by Pipeline', fontsize=16, fontweight='bold')
    plt.xlabel('Pipeline', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Metric', title_fontsize=12, fontsize=11, loc='best')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot timing metrics
    plt.figure(figsize=(12, 8))
    
    timing_df = df.melt(
        id_vars=['pipeline'],
        value_vars=['avg_detection_time', 'avg_classification_time', 'avg_total_time'],
        var_name='Timing Metric',
        value_name='Time (s)'
    )
    
    ax = sns.barplot(x='pipeline', y='Time (s)', hue='Timing Metric', data=timing_df,
                palette=sns.color_palette("Oranges_d", 3))
    plt.title('Timing Metrics by Pipeline', fontsize=16, fontweight='bold')
    plt.xlabel('Pipeline', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Metric', title_fontsize=12, fontsize=11, loc='best')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'timing_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot speed vs. accuracy with improved styling
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a colormap based on weighted F1 score
    norm = plt.Normalize(df['weighted_f1'].min(), df['weighted_f1'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    
    scatter = ax.scatter(
        df['avg_total_time'],
        df['accuracy'],
        s=150,
        alpha=0.8,
        c=df['weighted_f1'],
        cmap="viridis",
        edgecolors='white',
        linewidth=1.5
    )
    
    # Add pipeline names as labels
    for i, pipeline in enumerate(df['pipeline']):
        ax.annotate(
            pipeline,
            (df['avg_total_time'].iloc[i], df['accuracy'].iloc[i]),
            fontsize=11,
            ha='center',
            va='bottom',
            xytext=(0, 7),
            textcoords='offset points',
            fontweight='bold'
        )
    
    plt.title('Speed vs. Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Average Total Time (seconds)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Fix the colorbar issue by explicitly specifying the axes
    cbar = fig.colorbar(sm, ax=ax, label='Weighted F1 Score')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Weighted F1 Score', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'speed_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Plot confusion matrices for each pipeline with improved styling
    for pipeline in pipeline_names:
        pipeline_metrics = metrics[pipeline]
        
        if 'confusion_matrix' in pipeline_metrics:
            cm = np.array(pipeline_metrics['confusion_matrix']['matrix'])
            labels = pipeline_metrics['confusion_matrix']['labels']
            
            # Skip if there's only one class (handles the warning)
            if len(labels) <= 1:
                print(f"Skipping confusion matrix for {pipeline} - not enough classes")
                continue
            
            plt.figure(figsize=(10, 8))
            
            # Normalize confusion matrix for better visualization
            cm_sum = cm.sum(axis=1)
            # Avoid division by zero
            cm_sum[cm_sum == 0] = 1
            cm_norm = cm.astype('float') / cm_sum[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)  # Replace NaNs with zeros
            
            # Plot both normalized (colors) and absolute (text) values
            ax = sns.heatmap(
                cm_norm,
                annot=cm,
                fmt='d',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Normalized Frequency'}
            )
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
            
            plt.title(f'Confusion Matrix: {pipeline}', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted', fontsize=14)
            plt.ylabel('True', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'cm_{pipeline}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 5. NEW: F1 Score vs Efficiency Radar Chart
    if len(pipeline_names) > 1:
        # Number of variables
        categories = ['Accuracy', 'Macro F1', 'Weighted F1', 'Speed (1/time)']
        N = len(categories)
        
        # Calculate speed as inverse of time (higher is better)
        df['speed'] = 1.0 / (df['avg_total_time'] + 0.001)  # Add small constant to avoid division by zero
        
        # Normalize all metrics to [0,1] for radar chart
        df_radar = pd.DataFrame()
        df_radar['pipeline'] = df['pipeline']
        
        for col, new_col in zip(['accuracy', 'macro_f1', 'weighted_f1', 'speed'], categories):
            min_val = df[col].min()
            max_val = df[col].max()
            
            if max_val > min_val:
                df_radar[new_col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_radar[new_col] = df[col] / df[col]
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories, fontsize=14)
        
        # Draw the y-axis labels (0 to 1)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], fontsize=12)
        plt.ylim(0, 1)
        
        # Plot each pipeline
        for i, pipeline in enumerate(df_radar['pipeline']):
            values = df_radar.loc[df_radar['pipeline'] == pipeline, categories].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=pipeline, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Add title and legend
        plt.title('Performance Metrics Comparison', size=16, fontweight='bold', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Create a summary table
    summary_df = df[['pipeline', 'accuracy', 'macro_f1', 'weighted_f1', 'avg_total_time']]
    summary_df = summary_df.rename(columns={
        'accuracy': 'Accuracy',
        'macro_f1': 'Macro F1',
        'weighted_f1': 'Weighted F1',
        'avg_total_time': 'Avg. Time (s)'
    })
    
    # Sort by accuracy
    summary_df = summary_df.sort_values('Accuracy', ascending=False)
    
    # Save as CSV
    summary_df.to_csv(os.path.join(save_dir, 'summary_results.csv'), index=False)
    
    print(f"Visualizations saved to {save_dir}")


def visualize_detection(image: np.ndarray, prediction: Dict[str, Any], save_path: str = None):
    """
    Visualize a detection and classification result
    
    Args:
        image: Input image (RGB format, numpy array)
        prediction: Prediction dict from the pipeline
        save_path: Path to save visualization
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    
    # Set title with overall info
    class_name = prediction.get('main_vehicle_class', 'unknown')
    confidence = prediction.get('main_vehicle_confidence', 0.0)
    detection_time = prediction.get('detection_time', 0.0)
    classification_time = prediction.get('classification_time', 0.0)
    
    title = f"Vehicle: {class_name} (Conf: {confidence:.2f})\n"
    title += f"Detection: {detection_time:.3f}s | Classification: {classification_time:.3f}s"
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Extract main vehicle detection
    main_vehicle = prediction.get('main_vehicle')
    
    if main_vehicle:
        # Get bounding box
        bbox = main_vehicle.get('bbox', [])
        
        if len(bbox) == 4:
            # Create a rectangle patch
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box with a thicker, more visible line
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=3,
                edgecolor='lime',
                facecolor='none',
                alpha=0.8
            )
            plt.gca().add_patch(rect)
            
            # Add label with improved styling
            label_text = f"{class_name}: {confidence:.2f}"
            
            # Create a more prominent text box
            plt.text(
                x1,
                y1 - 10,
                label_text,
                color='white',
                fontsize=14,
                fontweight='bold',
                bbox=dict(
                    facecolor='green',
                    alpha=0.8,
                    boxstyle='round,pad=0.5',
                    edgecolor='black',
                    linewidth=1
                )
            )
    
    plt.axis('off')
    
    # Add a subtle border to the entire image
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['top'].set_color('gray')
    plt.gca().spines['right'].set_color('gray')
    plt.gca().spines['bottom'].set_color('gray')
    plt.gca().spines['left'].set_color('gray')
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()