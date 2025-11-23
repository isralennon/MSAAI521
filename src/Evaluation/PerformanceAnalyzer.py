"""
PerformanceAnalyzer: Analyze model performance across different scenarios.

This module provides detailed performance analysis including per-class metrics,
confidence threshold analysis, and error pattern identification.

Key Operations:
- Compute detailed per-class statistics
- Analyze performance across confidence thresholds
- Identify common failure modes
- Generate performance summary tables
"""

import numpy as np
from pathlib import Path


class PerformanceAnalyzer:
    """
    Analyzer for detailed model performance assessment.
    
    Provides tools for understanding model behavior beyond basic metrics,
    including analysis of failure cases and performance across different
    object types and scenarios.
    
    Attributes:
        class_names: List of detection class names
    """
    
    def __init__(self):
        """
        Initialize the performance analyzer.
        
        Technical Details:
            - Prepares analysis frameworks
            - Sets up metric tracking structures
        """
        self.class_names = ['Car', 'Truck/Bus', 'Pedestrian', 'Cyclist']
    
    def analyze_class_distribution(self, labels_dir):
        """
        Analyze class distribution in the dataset.
        
        Counts instances of each class across all labels to understand
        dataset balance and potential class imbalance issues.
        
        Args:
            labels_dir: Directory containing YOLO label files
        
        Returns:
            Dictionary mapping class names to instance counts
        
        Technical Details:
            - Parses all label files
            - Aggregates class counts
            - Identifies imbalanced classes
        """
        print("\nAnalyzing class distribution...")
        
        labels_dir = Path(labels_dir)
        class_counts = {name: 0 for name in self.class_names}
        
        # Count instances in each label file
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(self.class_names):
                            class_counts[self.class_names[class_id]] += 1
        
        # Print distribution
        total = sum(class_counts.values())
        print("\nClass Distribution:")
        print(f"  {'Class':<15} {'Count':<10} {'Percentage':<10}")
        print(f"  {'-'*40}")
        
        for class_name, count in class_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {class_name:<15} {count:<10} {percentage:>6.2f}%")
        
        print(f"\n  Total: {total}")
        
        return class_counts
    
    def compute_performance_summary(self, results):
        """
        Compute comprehensive performance summary.
        
        Extracts and organizes all key metrics from evaluation results
        into a structured summary for reporting or further analysis.
        
        Args:
            results: Evaluation results from ModelEvaluator
        
        Returns:
            Dictionary containing organized performance metrics
        
        Technical Details:
            - Extracts metrics from YOLO results object
            - Organizes by category (overall, per-class, speed)
            - Returns structured data for downstream use
        """
        summary = {
            'overall': {
                'mAP_50': float(results.box.map50),
                'mAP_50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr)
            },
            'per_class': {},
            'speed': {}
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            summary['per_class'][class_name] = {
                'mAP_50': float(results.box.maps[i]) if hasattr(results.box, 'maps') else 0.0,
                'precision': float(results.box.p[i]) if hasattr(results.box, 'p') else 0.0,
                'recall': float(results.box.r[i]) if hasattr(results.box, 'r') else 0.0
            }
        
        # Speed metrics
        if hasattr(results, 'speed'):
            summary['speed'] = {
                'preprocess_ms': results.speed.get('preprocess', 0),
                'inference_ms': results.speed.get('inference', 0),
                'postprocess_ms': results.speed.get('postprocess', 0),
                'total_ms': sum(results.speed.values()),
                'fps': 1000 / sum(results.speed.values()) if sum(results.speed.values()) > 0 else 0
            }
        
        return summary
    
    def print_summary(self, summary):
        """
        Print formatted performance summary.
        
        Args:
            summary: Performance summary dictionary from compute_performance_summary()
        
        Technical Details:
            - Pretty-prints structured metrics
            - Formatted for console display
        """
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        print("\nOverall Metrics:")
        for metric, value in summary['overall'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nPer-Class Performance:")
        for class_name, metrics in summary['per_class'].items():
            print(f"\n  {class_name}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
        
        if summary['speed']:
            print("\nInference Speed:")
            for metric, value in summary['speed'].items():
                if 'fps' in metric:
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value:.2f} ms")
        
        print("\n" + "="*80)

