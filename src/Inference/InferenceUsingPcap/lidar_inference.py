"""
LiDAR Inference with PCD File Support

This module extends the original MSAAI521 inference to support standard .pcd files
in addition to .pcd.bin (nuScenes binary format).

Supported formats:
- .pcd (Point Cloud Data - ASCII or binary, from Velodyne, Ouster, Open3D, etc.)
- .pcd.bin (nuScenes binary format)

Usage:
    from lidar_inference import LidarInferencePCD
    
    inference = LidarInferencePCD(model_path='path/to/model.pt')
    results = inference.predict_from_lidar_file('scan.pcd')  # or .pcd.bin
    inference.visualize_detections(results, save_path='output.png')
"""

import numpy as np
import cv2
from pathlib import Path
import struct
import sys

# Add the MSAAI521 source path for imports
MSAAI521_PATH = Path('/home/santoshaibox/msaai/AAI-521/final_project/MSAAI521')
sys.path.insert(0, str(MSAAI521_PATH))
sys.path.insert(0, str(MSAAI521_PATH / 'src'))

from ultralytics import YOLO
from Preprocessing.BEVRasterizer import BEVRasterizer

# Import GPU-accelerated rasterizer (PyTorch-based) from gpu_modules
try:
    from gpu_modules.bev_rasterizer_torch import BEVRasterizerTorch
    GPU_RASTERIZER_AVAILABLE = True
except ImportError:
    GPU_RASTERIZER_AVAILABLE = False
    BEVRasterizerTorch = None


class PCDReader:
    """
    Reader for standard .pcd (Point Cloud Data) files.
    
    Supports both ASCII and binary PCD formats as defined by the PCL library.
    """
    
    @staticmethod
    def read_pcd(file_path):
        """
        Read a .pcd file and return points as numpy array.
        
        Args:
            file_path: Path to .pcd file
        
        Returns:
            points: numpy array of shape (N, 4) with columns [x, y, z, intensity]
                   If intensity is not available, uses zeros.
        """
        file_path = Path(file_path)
        
        with open(file_path, 'rb') as f:
            # Parse header
            header = {}
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('DATA'):
                    data_type = line.split()[1].lower()
                    header['DATA'] = data_type
                    break
                elif line.startswith('#'):
                    continue
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        header[parts[0]] = parts[1:]
            
            # Extract field information
            fields = header.get('FIELDS', ['x', 'y', 'z'])
            sizes = [int(s) for s in header.get('SIZE', ['4'] * len(fields))]
            types = header.get('TYPE', ['F'] * len(fields))
            counts = [int(c) for c in header.get('COUNT', ['1'] * len(fields))]
            num_points = int(header.get('POINTS', ['0'])[0])
            width = int(header.get('WIDTH', ['0'])[0])
            height = int(header.get('HEIGHT', ['1'])[0])
            
            if num_points == 0:
                num_points = width * height
            
            # Build field map
            field_map = {}
            offset = 0
            for i, field in enumerate(fields):
                field_map[field.lower()] = {
                    'offset': offset,
                    'size': sizes[i],
                    'type': types[i],
                    'count': counts[i]
                }
                offset += sizes[i] * counts[i]
            
            point_size = offset
            
            # Find intensity field (various naming conventions)
            intensity_field = None
            for name in ['intensity', 'reflectivity', 'i', 'r', 'signal']:
                if name in field_map:
                    intensity_field = name
                    break
            
            if data_type == 'binary':
                # Read binary data
                raw_data = f.read()
                
                # Parse points
                points = []
                for i in range(num_points):
                    point_data = raw_data[i * point_size:(i + 1) * point_size]
                    if len(point_data) < point_size:
                        break
                    
                    # Extract x, y, z
                    x = PCDReader._unpack_field(point_data, field_map['x'])
                    y = PCDReader._unpack_field(point_data, field_map['y'])
                    z = PCDReader._unpack_field(point_data, field_map['z'])
                    
                    # Extract intensity if available
                    if intensity_field and intensity_field in field_map:
                        intensity = PCDReader._unpack_field(point_data, field_map[intensity_field])
                    else:
                        intensity = 0.0
                    
                    points.append([x, y, z, intensity])
                
                points = np.array(points, dtype=np.float32)
            
            elif data_type == 'binary_compressed':
                raise NotImplementedError("Compressed binary PCD not yet supported")
            
            else:  # ASCII
                points = []
                for line in f:
                    line = line.decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    values = line.split()
                    
                    # Get field indices
                    x_idx = fields.index('x') if 'x' in fields else 0
                    y_idx = fields.index('y') if 'y' in fields else 1
                    z_idx = fields.index('z') if 'z' in fields else 2
                    
                    x = float(values[x_idx])
                    y = float(values[y_idx])
                    z = float(values[z_idx])
                    
                    # Get intensity
                    intensity = 0.0
                    if intensity_field:
                        int_idx = fields.index(intensity_field)
                        if int_idx < len(values):
                            intensity = float(values[int_idx])
                    
                    points.append([x, y, z, intensity])
                
                points = np.array(points, dtype=np.float32)
        
        return points
    
    @staticmethod
    def _unpack_field(data, field_info):
        """Unpack a single field from binary data."""
        offset = field_info['offset']
        size = field_info['size']
        dtype = field_info['type']
        
        if dtype == 'F':
            if size == 4:
                return struct.unpack('<f', data[offset:offset + 4])[0]
            elif size == 8:
                return struct.unpack('<d', data[offset:offset + 8])[0]
        elif dtype == 'I':
            if size == 1:
                return struct.unpack('<B', data[offset:offset + 1])[0]
            elif size == 2:
                return struct.unpack('<H', data[offset:offset + 2])[0]
            elif size == 4:
                return struct.unpack('<I', data[offset:offset + 4])[0]
        elif dtype == 'U':
            if size == 1:
                return struct.unpack('<B', data[offset:offset + 1])[0]
            elif size == 2:
                return struct.unpack('<H', data[offset:offset + 2])[0]
            elif size == 4:
                return struct.unpack('<I', data[offset:offset + 4])[0]
        
        return 0.0


class PointCloudFilter:
    """
    Filter point clouds to region of interest.
    """
    
    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), z_range=(-3, 5)):
        """
        Initialize with spatial boundaries.
        
        Args:
            x_range: (min, max) forward/backward extent in meters
            y_range: (min, max) left/right extent in meters
            z_range: (min, max) height extent in meters
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
    
    def filter_points(self, points):
        """
        Filter points to ROI.
        
        Args:
            points: (4, N) or (N, 4) array of [x, y, z, intensity]
        
        Returns:
            Filtered points in (4, N) format
        """
        # Ensure (4, N) format
        if points.shape[0] != 4:
            points = points.T
        
        # Create boolean mask
        mask = (
            (points[0, :] >= self.x_range[0]) & (points[0, :] <= self.x_range[1]) &
            (points[1, :] >= self.y_range[0]) & (points[1, :] <= self.y_range[1]) &
            (points[2, :] >= self.z_range[0]) & (points[2, :] <= self.z_range[1])
        )
        
        return points[:, mask]


class LidarInferencePCD:
    """
    Inference engine for LiDAR-based BEV object detection.
    
    Supports both .pcd and .pcd.bin file formats.
    
    Attributes:
        model: Loaded YOLO model
        rasterizer: BEV rasterizer
        pc_filter: Point cloud filter
        class_names: Class name mapping
        class_colors: Color mapping for visualization
    """
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained YOLO weights (.pt file)
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
        """
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Check device
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Using device: {self.device}")
        if self.device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Target resolution for faster inference (100m range / 640 pixels = 0.15625)
        # Original training: 1280 pixels (0.078125 m/pixel)
        # Higher resolution for better detection
        target_resolution = 0.078125
        
        # Initialize preprocessing components
        self.pc_filter = PointCloudFilter()
        
        # Try to use GPU rasterizer (PyTorch-based), fallback to CPU if unavailable
        if GPU_RASTERIZER_AVAILABLE and torch.cuda.is_available():
            try:
                self.rasterizer = BEVRasterizerTorch(resolution=target_resolution)
                if self.rasterizer.use_gpu:
                    print(f"  Using GPU-accelerated BEV rasterization (PyTorch)")
                    print(f"  Expected BEV time: ~15ms (vs ~800ms on CPU)")
                else:
                    print(f"  GPU unavailable, using CPU BEV rasterization")
            except Exception as e:
                print(f"  GPU rasterizer failed ({e}), using CPU fallback")
                self.rasterizer = BEVRasterizer(resolution=target_resolution)
        else:
            self.rasterizer = BEVRasterizer(resolution=target_resolution)
            if not GPU_RASTERIZER_AVAILABLE:
                print(f"  Using CPU BEV rasterization (bev_rasterizer_torch.py not found)")
            else:
                print(f"  Using CPU BEV rasterization (CUDA not available)")
        
        # PCD reader
        self.pcd_reader = PCDReader()
        
        # Class mappings (matching MSAAI521 training)
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
        print(f"  Supported formats: .pcd, .pcd.bin")
    
    def load_lidar_from_file(self, lidar_file_path):
        """
        Load LiDAR point cloud from file (auto-detects format).
        
        Args:
            lidar_file_path: Path to LiDAR file (.pcd or .pcd.bin)
        
        Returns:
            points: numpy array of shape (N, 4) with columns [x, y, z, intensity]
        """
        lidar_file_path = Path(lidar_file_path)
        
        if not lidar_file_path.exists():
            raise FileNotFoundError(f"LiDAR file not found: {lidar_file_path}")
        
        suffix = ''.join(lidar_file_path.suffixes).lower()
        
        if suffix == '.pcd.bin':
            # nuScenes binary format (5 floats per point)
            points = np.fromfile(str(lidar_file_path), dtype=np.float32).reshape(-1, 5)
            # Drop ring index, keep [x, y, z, intensity]
            points = points[:, :4]
            print(f"Loaded {points.shape[0]} points from .pcd.bin file")
        
        elif suffix == '.pcd':
            # Standard PCD format
            points = self.pcd_reader.read_pcd(lidar_file_path)
            print(f"Loaded {points.shape[0]} points from .pcd file")
        
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .pcd or .pcd.bin")
        
        return points
    
    def preprocess_lidar(self, points):
        """
        Preprocess LiDAR points to BEV image.
        
        Args:
            points: Point cloud array (N, 4) - [x, y, z, intensity]
        
        Returns:
            bev_image: BEV image as numpy array (H, W, 3)
        """
        import torch
        
        # Convert to (4, N) format and filter
        if points.shape[1] == 4:
            points = points.T
        
        # Use GPU for filtering if available
        if self.device == 'cuda':
            # Move to GPU for faster filtering
            points_gpu = torch.from_numpy(points).cuda()
            
            # Filter on GPU
            filtered_points_gpu = self.pc_filter.filter_points(points_gpu.cpu().numpy())
            filtered_points = filtered_points_gpu
        else:
            filtered_points = self.pc_filter.filter_points(points)
        
        # Rasterize to BEV image (this is still CPU-bound)
        bev_image = self.rasterizer.rasterize(filtered_points)
        
        print(f"Generated BEV image: {bev_image.shape} from {filtered_points.shape[1]} points")
        return bev_image
    
    def predict(self, bev_image):
        """
        Run YOLO detection on BEV image.
        
        Args:
            bev_image: BEV image (H, W, 3)
        
        Returns:
            results: YOLO results object
        """
        results = self.model.predict(
            bev_image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        num_detections = len(results[0].boxes) if results else 0
        print(f"Detected {num_detections} objects")
        
        return results[0]
    
    def predict_from_lidar_file(self, lidar_file_path):
        """
        End-to-end inference: LiDAR file → detections.
        
        Args:
            lidar_file_path: Path to LiDAR file (.pcd or .pcd.bin)
        
        Returns:
            Dictionary containing:
                - bev_image: Generated BEV image
                - results: YOLO detection results
                - detections: List of detection dictionaries
                - points: Original point cloud (N, 4)
        """
        # Load LiDAR data
        points = self.load_lidar_from_file(lidar_file_path)
        
        # Preprocess to BEV
        bev_image = self.preprocess_lidar(points)
        
        # Run detection
        results = self.predict(bev_image)
        
        # Parse detections
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'class_id': cls,
                    'class_name': self.class_names.get(cls, f'class_{cls}'),
                    'confidence': conf
                })
        
        return {
            'bev_image': bev_image,
            'results': results,
            'detections': detections,
            'points': points
        }
    
    def visualize_detections(self, prediction_result, save_path=None, show=False):
        """
        Visualize detections by drawing bounding boxes on BEV image.
        
        Args:
            prediction_result: Output from predict_from_lidar_file()
            save_path: Optional path to save visualization
            show: If True, display image in window
        
        Returns:
            annotated_image: BEV image with bounding boxes
        """
        bev_image = prediction_result['bev_image'].copy()
        detections = prediction_result['detections']
        
        # Convert to BGR if grayscale
        if len(bev_image.shape) == 2:
            bev_image = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            cls = det['class_id']
            conf = det['confidence']
            
            color = self.class_colors.get(cls, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(bev_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class_name']}: {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                bev_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color, -1
            )
            cv2.putText(
                bev_image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        
        # Add detection count
        summary = f"Detections: {len(detections)}"
        cv2.putText(
            bev_image, summary, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )
        
        # Add ego vehicle marker
        h, w = bev_image.shape[:2]
        center = (w // 2, h // 2)
        cv2.circle(bev_image, center, 10, (0, 255, 255), -1)
        cv2.putText(bev_image, "EGO", (center[0] - 20, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw distance rings
        for radius_m in [20, 40]:
            radius_px = int(radius_m / self.rasterizer.resolution)
            cv2.circle(bev_image, center, radius_px, (100, 100, 100), 1)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), bev_image)
            print(f"✓ Saved visualization to: {save_path}")
        
        if show:
            cv2.imshow('BEV Detections', bev_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return bev_image
    
    def pixel_to_meters(self, pixel_coords):
        """
        Convert pixel coordinates to metric coordinates (meters).
        
        Args:
            pixel_coords: (x_pixel, y_pixel)
        
        Returns:
            (x_meters, y_meters) in vehicle coordinate frame
        """
        pixel_x, pixel_y = pixel_coords
        
        center_x = self.rasterizer.width / 2
        center_y = self.rasterizer.height / 2
        
        dx_pixels = pixel_x - center_x
        dy_pixels = -(pixel_y - center_y)
        
        x_meters = dy_pixels * self.rasterizer.resolution
        y_meters = -dx_pixels * self.rasterizer.resolution
        
        return (x_meters, y_meters)
    
    def get_detection_in_meters(self, detection):
        """
        Convert detection box from pixels to meters.
        
        Args:
            detection: Detection dictionary with 'box' in pixel coordinates
        
        Returns:
            Dictionary with center and size in meters
        """
        x1, y1, x2, y2 = detection['box']
        
        center_x_px = (x1 + x2) / 2
        center_y_px = (y1 + y2) / 2
        width_px = x2 - x1
        height_px = y2 - y1
        
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
    Example usage and command-line interface.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LiDAR Inference with PCD Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference on .pcd file
  python lidar_inference.py --model /path/to/model.pt --lidar scan.pcd --output result.png
  
  # Inference on .pcd.bin file (nuScenes)
  python lidar_inference.py --model model.pt --lidar scan.pcd.bin --output result.png
  
  # Adjust confidence threshold
  python lidar_inference.py --model model.pt --lidar scan.pcd --conf 0.3
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--lidar', type=str, required=True,
                        help='Path to LiDAR file (.pcd or .pcd.bin)')
    parser.add_argument('--output', type=str, default='detection_output.png',
                        help='Output visualization path')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--show', action='store_true',
                        help='Display result window')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not Path(args.model).exists():
        print(f"✗ Model not found: {args.model}")
        sys.exit(1)
    
    if not Path(args.lidar).exists():
        print(f"✗ LiDAR file not found: {args.lidar}")
        sys.exit(1)
    
    # Initialize inference engine
    inference = LidarInferencePCD(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run inference
    print(f"\nRunning inference on: {args.lidar}")
    results = inference.predict_from_lidar_file(args.lidar)
    
    # Print detections
    print(f"\n{'='*60}")
    print("DETECTIONS (Vehicle Frame - Meters)")
    print(f"{'='*60}")
    
    for i, det in enumerate(results['detections'], 1):
        det_meters = inference.get_detection_in_meters(det)
        x, y = det_meters['center']
        w, h = det_meters['size']
        distance = (x**2 + y**2)**0.5
        
        print(f"\n{i}. {det_meters['class'].upper()} (conf: {det_meters['confidence']:.2f})")
        print(f"   Position: x={x:6.2f}m (forward), y={y:6.2f}m (left)")
        print(f"   Size:     {w:5.2f}m × {h:5.2f}m")
        print(f"   Distance: {distance:6.2f}m from vehicle")
    
    if len(results['detections']) == 0:
        print("\nNo objects detected.")
        print("Try lowering the confidence threshold with --conf 0.15")
    
    # Visualize
    print(f"\n{'='*60}")
    inference.visualize_detections(results, save_path=args.output, show=args.show)
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
