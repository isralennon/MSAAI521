"""
LidarInference: Real-time inference on LiDAR data using trained BEV detection model.

This module provides end-to-end inference pipeline:
1. Load raw LiDAR point cloud data (.pcd.bin format)
2. Preprocess to BEV image (same as training pipeline)
3. Run YOLO detection on BEV image
4. Draw bounding boxes on BEV visualization
5. Optionally project boxes back to 3D space

Usage:
    from Inference.LidarInference import LidarInference
    
    # Initialize with trained model
    inference = LidarInference(model_path='build/runs/detect/stage2_finetune/weights/best.pt')
    
    # Run inference on single LiDAR scan
    results = inference.predict_from_lidar_file('path/to/lidar.pcd.bin')
    
    # Visualize results
    inference.visualize_detections(results, save_path='output.png')
"""

import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from Preprocessing.PointCloudProcessor import PointCloudProcessor
from Preprocessing.BEVRasterizer import BEVRasterizer


class LidarInference:
    """
    Inference engine for LiDAR-based BEV object detection.
    
    This class handles the complete inference pipeline from raw LiDAR point clouds
    to detected bounding boxes, matching the preprocessing used during training.
    
    Attributes:
        model: Loaded YOLO model
        pc_processor: Point cloud processor (matches training)
        rasterizer: BEV rasterizer (matches training)
        class_names: Class name mapping
        class_colors: Color mapping for visualization
    """
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the inference engine with a trained model.
        
        Args:
            model_path: Path to trained YOLO weights (.pt file)
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
        
        Technical Details:
            - Loads the trained model
            - Initializes preprocessing components with same parameters as training
            - Sets up class mappings and visualization colors
        """
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Confidence and IoU thresholds
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize preprocessing components (must match training configuration)
        # For 1280x1280 images: 100m range / 1280 pixels = 0.078125 m/pixel
        target_resolution = 0.078125
        
        # Point cloud processor - handles loading and filtering LiDAR data
        # Note: nusc parameter is None for standalone inference
        self.pc_processor = PointCloudProcessor(nusc=None)
        
        # BEV rasterizer - converts 3D points to 2D BEV image
        self.rasterizer = BEVRasterizer(resolution=target_resolution)
        
        # Class mappings (must match training)
        self.class_names = {
            0: 'car',
            1: 'truck_bus',
            2: 'pedestrian',
            3: 'cyclist'
        }
        
        # Colors for visualization (BGR format for OpenCV)
        self.class_colors = {
            0: (0, 255, 0),      # Car: Green
            1: (255, 0, 0),      # Truck/Bus: Blue
            2: (0, 0, 255),      # Pedestrian: Red
            3: (255, 255, 0)     # Cyclist: Cyan
        }
        
        print(f"✓ Model loaded successfully")
        print(f"  Confidence threshold: {conf_threshold}")
        print(f"  IoU threshold: {iou_threshold}")
        print(f"  BEV resolution: {target_resolution:.6f} m/pixel")
        print(f"  Image size: {self.rasterizer.width}x{self.rasterizer.height}")
    
    def load_lidar_from_file(self, lidar_file_path):
        """
        Load raw LiDAR point cloud from .pcd.bin file.
        
        Args:
            lidar_file_path: Path to LiDAR file (.pcd.bin format)
        
        Returns:
            points: Numpy array of shape (N, 5) with columns [x, y, z, intensity, ring]
        
        Technical Details:
            - nuScenes format: Each point has 5 values (x, y, z, intensity, ring_index)
            - Binary format: little-endian float32
            - Coordinate system: x=forward, y=left, z=up (LiDAR frame)
        """
        lidar_file_path = Path(lidar_file_path)
        if not lidar_file_path.exists():
            raise FileNotFoundError(f"LiDAR file not found: {lidar_file_path}")
        
        # Load binary point cloud (nuScenes format: 5 floats per point)
        points = np.fromfile(str(lidar_file_path), dtype=np.float32).reshape(-1, 5)
        
        print(f"Loaded {points.shape[0]} points from {lidar_file_path.name}")
        return points
    
    def preprocess_lidar(self, points):
        """
        Preprocess LiDAR points to BEV image (matches training preprocessing).
        
        Args:
            points: Raw point cloud array (N, 5) - [x, y, z, intensity, ring]
        
        Returns:
            bev_image: BEV image as numpy array (H, W, 3) - ready for YOLO inference
        
        Pipeline:
            1. Filter points to region of interest (ROI)
            2. Rasterize to BEV image (height, intensity, density channels)
            3. Return image in format expected by YOLO
        """
        # Convert from (N, 5) to (4, N) format expected by filter_points
        # Select only [x, y, z, intensity], drop ring index
        points_transposed = points[:, :4].T  # (N, 4) -> (4, N)
        
        # Filter points to region of interest (removes distant/irrelevant points)
        filtered_points = self.pc_processor.filter_points(points_transposed)
        
        # Rasterize to BEV image
        bev_image = self.rasterizer.rasterize(filtered_points)
        
        print(f"Generated BEV image: {bev_image.shape}")
        return bev_image
    
    def predict(self, bev_image):
        """
        Run YOLO detection on BEV image.
        
        Args:
            bev_image: BEV image (H, W, 3) - preprocessed point cloud
        
        Returns:
            results: YOLO results object containing detections
        
        Technical Details:
            - Runs model inference with configured thresholds
            - Returns Ultralytics Results object with boxes, scores, classes
            - Boxes are in pixel coordinates (can be converted back to meters)
        """
        # Run inference
        results = self.model.predict(
            bev_image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Extract detection info
        num_detections = len(results[0].boxes) if results else 0
        print(f"Detected {num_detections} objects")
        
        return results[0]  # Return first result (single image)
    
    def predict_from_lidar_file(self, lidar_file_path):
        """
        End-to-end inference: LiDAR file → detections.
        
        Args:
            lidar_file_path: Path to LiDAR .pcd.bin file
        
        Returns:
            Dictionary containing:
                - bev_image: Generated BEV image
                - results: YOLO detection results
                - detections: List of detection dictionaries with box, class, conf
        
        Usage:
            results = inference.predict_from_lidar_file('lidar_scan.pcd.bin')
            for det in results['detections']:
                print(f"Class: {det['class_name']}, Confidence: {det['confidence']:.2f}")
        """
        # Load LiDAR data
        points = self.load_lidar_from_file(lidar_file_path)
        
        # Preprocess to BEV
        bev_image = self.preprocess_lidar(points)
        
        # Run detection
        results = self.predict(bev_image)
        
        # Parse detections into structured format
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'box': [x1, y1, x2, y2],  # Pixel coordinates
                    'class_id': cls,
                    'class_name': self.class_names[cls],
                    'confidence': conf
                })
        
        return {
            'bev_image': bev_image,
            'results': results,
            'detections': detections
        }
    
    def visualize_detections(self, prediction_result, save_path=None, show=False):
        """
        Visualize detections by drawing bounding boxes on BEV image.
        
        Args:
            prediction_result: Output from predict_from_lidar_file()
            save_path: Optional path to save visualization (e.g., 'output.png')
            show: If True, display image in window (requires GUI)
        
        Returns:
            annotated_image: BEV image with drawn bounding boxes
        
        Visualization Details:
            - Boxes drawn with class-specific colors
            - Labels show: class name + confidence score
            - Grid overlay shows scale (optional)
        """
        # Get BEV image and detections
        bev_image = prediction_result['bev_image'].copy()
        detections = prediction_result['detections']
        
        # Convert to BGR for OpenCV drawing (if grayscale)
        if len(bev_image.shape) == 2:
            bev_image = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            cls = det['class_id']
            conf = det['confidence']
            
            # Get color for this class
            color = self.class_colors.get(cls, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(bev_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{det['class_name']}: {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                bev_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                bev_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1
            )
        
        # Add detection count
        summary = f"Detections: {len(detections)}"
        cv2.putText(
            bev_image,
            summary,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), bev_image)
            print(f"✓ Saved visualization to: {save_path}")
        
        # Show if requested
        if show:
            cv2.imshow('BEV Detections', bev_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return bev_image
    
    def pixel_to_meters(self, pixel_coords):
        """
        Convert pixel coordinates to metric coordinates (meters).
        
        Args:
            pixel_coords: (x_pixel, y_pixel) or array of coordinates
        
        Returns:
            metric_coords: (x_meters, y_meters) in vehicle coordinate frame
        
        Technical Details:
            - Origin is at center of BEV image (vehicle position)
            - x-axis: forward (top of image = +x)
            - y-axis: left (left of image = +y)
        """
        pixel_x, pixel_y = pixel_coords
        
        # Convert from image coordinates to metric coordinates
        # Image origin (0,0) is top-left, metric origin is center
        center_x = self.rasterizer.width / 2
        center_y = self.rasterizer.height / 2
        
        # Pixel offset from center
        dx_pixels = pixel_x - center_x
        dy_pixels = -(pixel_y - center_y)  # Flip y-axis (image y goes down)
        
        # Convert to meters
        x_meters = dy_pixels * self.rasterizer.resolution  # Image y → vehicle x
        y_meters = -dx_pixels * self.rasterizer.resolution  # Image x → vehicle y
        
        return (x_meters, y_meters)
    
    def get_detection_in_meters(self, detection):
        """
        Convert detection box from pixels to meters (vehicle frame).
        
        Args:
            detection: Detection dictionary with 'box' in pixel coordinates
        
        Returns:
            Dictionary with box in meters: {center: (x, y), size: (width, height)}
        """
        x1, y1, x2, y2 = detection['box']
        
        # Box center in pixels
        center_x_px = (x1 + x2) / 2
        center_y_px = (y1 + y2) / 2
        
        # Box size in pixels
        width_px = x2 - x1
        height_px = y2 - y1
        
        # Convert to meters
        center_meters = self.pixel_to_meters((center_x_px, center_y_px))
        width_meters = width_px * self.rasterizer.resolution
        height_meters = height_px * self.rasterizer.resolution
        
        return {
            'center': center_meters,
            'size': (width_meters, height_meters),
            'class': detection['class_name'],
            'confidence': detection['confidence']
        }


def main():
    """
    Example usage of LidarInference.
    
    Run this script directly to test inference on a sample LiDAR file.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on LiDAR data')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--lidar', type=str, required=True,
                        help='Path to LiDAR file (.pcd.bin)')
    parser.add_argument('--output', type=str, default='detection_output.png',
                        help='Output visualization path')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--show', action='store_true',
                        help='Display result window')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = LidarInference(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run inference
    print(f"\nRunning inference on: {args.lidar}")
    results = inference.predict_from_lidar_file(args.lidar)
    
    # Print detections in metric coordinates
    print(f"\n{'='*60}")
    print("DETECTIONS (Vehicle Frame - Meters)")
    print(f"{'='*60}")
    for i, det in enumerate(results['detections'], 1):
        det_meters = inference.get_detection_in_meters(det)
        x, y = det_meters['center']
        w, h = det_meters['size']
        print(f"\n{i}. {det_meters['class'].upper()} (conf: {det_meters['confidence']:.2f})")
        print(f"   Center: x={x:6.2f}m, y={y:6.2f}m")
        print(f"   Size:   w={w:6.2f}m, h={h:6.2f}m")
    
    if len(results['detections']) == 0:
        print("\nNo objects detected.")
    
    # Visualize
    print(f"\n{'='*60}")
    inference.visualize_detections(results, save_path=args.output, show=args.show)
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
