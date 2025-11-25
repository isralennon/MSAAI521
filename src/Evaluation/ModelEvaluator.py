"""
ModelEvaluator: Evaluate trained YOLO model on test set.

This module handles model evaluation including inference on test set, metrics
computation (mAP, precision, recall), and performance analysis. It provides
comprehensive assessment of detection performance.

Key Operations:
- Run inference on test set
- Compute detection metrics (mAP@0.5, mAP@0.5:0.95)
- Calculate per-class performance
- Measure inference speed
"""

from ultralytics import YOLO
from pathlib import Path


class ModelEvaluator:
    """
    Evaluator for trained YOLO detection models.
    
    This class handles model evaluation on test data, computing standard object
    detection metrics and analyzing performance across different classes and
    confidence thresholds.
    
    Attributes:
        model: YOLO model instance loaded from weights
        model_path: Path to model weights file
        dataset_yaml: Path to dataset configuration
        class_names: List of detection class names
    """
    
    def __init__(self, model_path, dataset_yaml):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to trained model weights (.pt file)
            dataset_yaml: Path to dataset.yaml configuration
        
        Technical Details:
            - Loads model from weights file
            - Validates model architecture
            - Prepares for evaluation on test set
        """
        self.model_path = Path(model_path)
        self.dataset_yaml = str(dataset_yaml)
        self.class_names = ['Car', 'Truck/Bus', 'Pedestrian', 'Cyclist']
        
        # Load trained model
        print(f"\nLoading model from: {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        print("✓ Model loaded successfully")
    
    def evaluate(self, conf_threshold=0.25, iou_threshold=0.45, img_size=1024):
        """
        Evaluate model on test set.
        
        Runs inference on all test images and computes comprehensive detection
        metrics including mAP, precision, recall, and per-class performance.
        
        Args:
            conf_threshold: Confidence threshold for detections (default: 0.25)
            iou_threshold: IoU threshold for NMS (default: 0.45)
            img_size: Input image size (default: 1024 to match BEV resolution)
        
        Returns:
            Results object containing metrics and predictions
        
        Technical Details:
            
            **Metrics Computed:**
            
            • mAP@0.5 (Mean Average Precision at IoU=0.5):
              - Standard COCO metric for loose localization
              - Considers detection "correct" if IoU ≥ 0.5
              - Values typically 0.5-0.9 for good detectors
            
            • mAP@0.5:0.95 (Mean Average Precision at IoU=0.5:0.95):
              - Averaged over IoU thresholds from 0.5 to 0.95 (step 0.05)
              - More stringent metric requiring tighter localization
              - Values typically 0.3-0.6 for good detectors
            
            • Precision:
              - TP / (TP + FP)
              - Proportion of correct detections among all detections
              - Higher = fewer false alarms
            
            • Recall:
              - TP / (TP + FN)
              - Proportion of ground truth objects detected
              - Higher = fewer missed objects
            
            **Confidence Threshold:**
            - 0.25: Default, balances precision and recall
            - Lower: More detections, higher recall, lower precision
            - Higher: Fewer detections, lower recall, higher precision
            
            **NMS IoU Threshold:**
            - 0.45: Removes overlapping boxes (keep best)
            - Lower: More aggressive suppression
            - Higher: Keeps more overlapping detections
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        print(f"\nEvaluation settings:")
        print(f"  Confidence threshold: {conf_threshold}")
        print(f"  NMS IoU threshold: {iou_threshold}")
        print(f"  Image size: {img_size}")
        
        # ============================================================
        # Run validation on test split
        # ============================================================
        print(f"\nRunning inference on test set...")
        
        results = self.model.val(
            data=self.dataset_yaml,
            split='test',
            imgsz=img_size,
            batch=4,
            conf=conf_threshold,
            iou=iou_threshold,
            plots=True,
            save_json=True,
            save_txt=True
        )
        
        print("✓ Evaluation complete")
        
        return results
    
    def print_metrics(self, results):
        """
        Print comprehensive evaluation metrics.
        
        Displays overall and per-class performance metrics in a formatted
        table for easy interpretation.
        
        Args:
            results: Results object from evaluate()
        
        Technical Details:
            - Extracts metrics from YOLO results object
            - Formats for console display
            - Includes inference timing statistics
        """
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        
        # ============================================================
        # Overall metrics
        # ============================================================
        print(f"\nOverall Performance:")
        print(f"  mAP@0.5:      {results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"  Precision:    {results.box.mp:.4f}")
        print(f"  Recall:       {results.box.mr:.4f}")
        
        # ============================================================
        # Per-class metrics
        # ============================================================
        print(f"\nPer-Class Performance:")
        print(f"  {'Class':<15} {'mAP@0.5':<10} {'Precision':<12} {'Recall':<10}")
        print(f"  {'-'*50}")
        
        for i, class_name in enumerate(self.class_names):
            map50 = results.box.maps[i] if hasattr(results.box, 'maps') else 0.0
            precision = results.box.p[i] if hasattr(results.box, 'p') else 0.0
            recall = results.box.r[i] if hasattr(results.box, 'r') else 0.0
            
            print(f"  {class_name:<15} {map50:<10.4f} {precision:<12.4f} {recall:<10.4f}")
        
        # ============================================================
        # Inference speed
        # ============================================================
        print(f"\nInference Speed:")
        if hasattr(results, 'speed'):
            preprocess_time = results.speed.get('preprocess', 0)
            inference_time = results.speed.get('inference', 0)
            postprocess_time = results.speed.get('postprocess', 0)
            total_time = preprocess_time + inference_time + postprocess_time
            
            print(f"  Preprocess:  {preprocess_time:.2f} ms")
            print(f"  Inference:   {inference_time:.2f} ms")
            print(f"  Postprocess: {postprocess_time:.2f} ms")
            print(f"  Total:       {total_time:.2f} ms")
            print(f"  FPS:         {1000 / total_time:.2f}")
        
        print("\n" + "="*80)
    
    def predict_batch(self, image_paths, conf_threshold=0.25, save_dir=None):
        """
        Run inference on a batch of images.
        
        Useful for visualizing predictions on specific images or creating
        demo outputs.
        
        Args:
            image_paths: List of paths to images
            conf_threshold: Confidence threshold (default: 0.25)
            save_dir: Directory to save prediction visualizations (optional)
        
        Returns:
            List of prediction results, one per image
        
        Technical Details:
            - Processes images in batch for efficiency
            - Optionally saves annotated images
            - Returns raw prediction results for further processing
        """
        results = self.model.predict(
            source=image_paths,
            conf=conf_threshold,
            save=save_dir is not None,
            project=save_dir,
            exist_ok=True
        )
        
        return results

