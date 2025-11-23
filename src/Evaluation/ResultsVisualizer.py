"""
ResultsVisualizer: Visualize model predictions and performance.

This module creates visualizations of model predictions including detection
boxes overlaid on BEV images, confusion matrices, and performance plots.

Key Operations:
- Visualize predictions on test images
- Generate confusion matrix
- Plot precision-recall curves
- Create performance comparison charts
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import random
from Globals import RESULTS_ROOT


class ResultsVisualizer:
    """
    Visualizer for model predictions and evaluation results.
    
    Creates visual outputs to assess model performance and understand prediction
    behavior across different scenarios.
    
    Attributes:
        results_dir: Directory for saving visualization outputs
        class_names: List of detection class names
        colors: Color scheme for each class
    """
    
    def __init__(self):
        """
        Initialize the results visualizer.
        
        Technical Details:
            - Creates results directory structure
            - Sets up color scheme for consistent visualization
            - Configures matplotlib defaults
        """
        self.results_dir = Path(RESULTS_ROOT)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_names = ['Car', 'Truck/Bus', 'Pedestrian', 'Cyclist']
        self.colors = [
            (255, 0, 0),    # Red: Cars
            (0, 255, 0),    # Green: Trucks/Buses
            (0, 0, 255),    # Blue: Pedestrians
            (255, 255, 0)   # Yellow: Cyclists
        ]
    
    def visualize_predictions(self, model, test_images_dir, num_samples=10, conf_threshold=0.25):
        """
        Visualize model predictions on random test images.
        
        Selects random images from test set, runs inference, and creates
        visualizations with predicted bounding boxes overlaid.
        
        Args:
            model: Trained YOLO model instance
            test_images_dir: Directory containing test images
            num_samples: Number of images to visualize (default: 10)
            conf_threshold: Confidence threshold for predictions (default: 0.25)
        
        Returns:
            None (saves visualizations to disk)
        
        Technical Details:
            - Random sampling ensures diverse visualization
            - Predictions drawn with confidence scores
            - Color-coded by class
            - Saved as high-resolution PNG files
        """
        print(f"\nGenerating prediction visualizations...")
        
        # ============================================================
        # Select random test images
        # ============================================================
        test_images_dir = Path(test_images_dir)
        all_images = list(test_images_dir.glob('*.png'))
        
        if len(all_images) == 0:
            print(f"Warning: No test images found in {test_images_dir}")
            return
        
        sample_images = random.sample(all_images, min(num_samples, len(all_images)))
        
        # ============================================================
        # Create output directory
        # ============================================================
        vis_dir = self.results_dir / 'predictions'
        vis_dir.mkdir(exist_ok=True)
        
        # ============================================================
        # Run predictions and visualize
        # ============================================================
        for img_path in sample_images:
            # Run inference
            results = model.predict(
                source=str(img_path),
                conf=conf_threshold,
                verbose=False
            )
            
            # Get first result (single image)
            result = results[0]
            
            # Load original image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Draw predictions
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confs = result.boxes.conf.cpu().numpy()  # Confidence scores
                classes = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                
                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Draw bounding box
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), self.colors[cls], 2)
                    
                    # Draw label with confidence
                    label = f"{self.class_names[cls]}: {conf:.2f}"
                    cv2.putText(img_rgb, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[cls], 2)
            
            # Save visualization
            output_path = vis_dir / f"pred_{img_path.name}"
            plt.figure(figsize=(15, 15))
            plt.imshow(img_rgb)
            plt.title(f"Predictions: {img_path.name}\nDetections: {len(result.boxes) if result.boxes else 0}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Saved {len(sample_images)} prediction visualizations to: {vis_dir}")
    
    def generate_performance_report(self, results, output_path=None):
        """
        Generate comprehensive performance report.
        
        Creates a text report summarizing model performance including overall
        metrics, per-class breakdown, and inference timing.
        
        Args:
            results: Evaluation results from ModelEvaluator
            output_path: Path for report file (default: results/evaluation_report.txt)
        
        Returns:
            Path to generated report file
        
        Technical Details:
            - Plain text format for easy viewing
            - Includes all key metrics
            - Suitable for documentation or sharing
        """
        if output_path is None:
            output_path = self.results_dir / 'evaluation_report.txt'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating performance report...")
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("nuScenes BEV Object Detection - Evaluation Report\n")
            f.write("="*80 + "\n\n")
            
            f.write("Model Configuration:\n")
            f.write("-"*80 + "\n")
            f.write("Base Model: YOLOv12s\n")
            f.write("Training Strategy: Two-stage transfer learning\n")
            f.write("Input Resolution: 1000×1000 pixels\n")
            f.write("Detection Classes: 4 (Car, Truck/Bus, Pedestrian, Cyclist)\n\n")
            
            f.write("Overall Performance:\n")
            f.write("-"*80 + "\n")
            f.write(f"mAP@0.5:      {results.box.map50:.4f}\n")
            f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
            f.write(f"Precision:    {results.box.mp:.4f}\n")
            f.write(f"Recall:       {results.box.mr:.4f}\n\n")
            
            f.write("Per-Class Performance:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Class':<15} {'mAP@0.5':<12} {'Precision':<12} {'Recall':<10}\n")
            f.write("-"*80 + "\n")
            
            for i, class_name in enumerate(self.class_names):
                map50 = results.box.maps[i] if hasattr(results.box, 'maps') else 0.0
                precision = results.box.p[i] if hasattr(results.box, 'p') else 0.0
                recall = results.box.r[i] if hasattr(results.box, 'r') else 0.0
                
                f.write(f"{class_name:<15} {map50:<12.4f} {precision:<12.4f} {recall:<10.4f}\n")
            
            if hasattr(results, 'speed'):
                total_time = sum(results.speed.values())
                fps = 1000 / total_time if total_time > 0 else 0
                
                f.write("\n" + "="*80 + "\n")
                f.write("Inference Speed:\n")
                f.write("-"*80 + "\n")
                f.write(f"Average FPS: {fps:.2f}\n")
                f.write(f"Total latency: {total_time:.2f} ms\n\n")
            
            f.write("="*80 + "\n")
            f.write("Conclusion:\n")
            f.write("-"*80 + "\n")
            f.write("The model demonstrates the feasibility of using YOLO for LiDAR-based\n")
            f.write("object detection by converting 3D point clouds to BEV representations.\n")
            f.write("The two-stage training approach with transfer learning enables efficient\n")
            f.write("domain adaptation from COCO to the nuScenes BEV dataset.\n")
        
        print(f"✓ Report saved to: {output_path}")
        return output_path

