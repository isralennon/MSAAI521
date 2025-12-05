"""
Ouster SDK Visualization with Object Detection

Uses Ouster's native PointViz for professional point cloud rendering
with our YOLO detection bounding boxes overlaid.

Supports both PCAP and OSF file formats.

Usage:
    # PCAP file (requires separate JSON metadata)
    python3 pcap_visualizer_ouster.py \
        --source Urban_Drive_.pcap \
        --json Urban_Drive_.json \
        --model /path/to/model.pt
    
    # OSF file (metadata embedded)
    python3 pcap_visualizer_ouster.py \
        --source Urban_Drive_.osf \
        --model /path/to/model.pt

Controls (Ouster default):
    SPACE   - Pause/Resume
    .       - Step forward (when paused)
    ,       - Step backward (when paused)
    0-6     - Change point cloud coloring mode
    p       - Toggle point cloud visibility
    b       - Toggle bounding boxes (our addition)
    ESC/Q   - Quit
    
    Mouse:
      Left drag   - Rotate view
      Right drag  - Pan
      Scroll      - Zoom
"""

import numpy as np
from pathlib import Path
import argparse
import sys
import warnings
import time
from collections import defaultdict

warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    from ouster.sdk import open_source, viz
    from ouster.sdk.core import SensorInfo, XYZLut, ChanField
except ImportError as e:
    print(f"Error: ouster-sdk not installed or missing components: {e}")
    print("Install with: pip install ouster-sdk")
    sys.exit(1)

from lidar_inference import LidarInferencePCD


class OusterDetectionViz:
    """
    Ouster SDK visualization with integrated object detection.
    Uses Ouster's professional PointViz with detection boxes overlaid.
    """
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45, detection_skip=1):
        """Initialize the visualizer with detection model."""
        self.inference = LidarInferencePCD(model_path, conf_threshold, iou_threshold)
        self.detection_skip = detection_skip  # Run detection every N frames
        
        # Class colors (RGBA 0-1) - reduced alpha for transparency
        self.class_colors = {
            0: (0.0, 1.0, 0.0, 0.3),    # car - green
            1: (0.0, 0.0, 1.0, 0.3),    # truck_bus - blue
            2: (1.0, 0.0, 0.0, 0.3),    # pedestrian - red
            3: (0.0, 1.0, 1.0, 0.3),    # cyclist - cyan
        }
        
        self.show_boxes = True
        self.current_frame = 0
        self.stop_processing = False
        self.paused = False
        self.latest_detections = []
        
        print(f"\n{'='*70}")
        print("Ouster SDK Visualization with Object Detection")
        print(f"{'='*70}")
        print("\nControls:")
        print("  SPACE       - Pause/Resume")
        print("  0-9         - Cycle through coloring modes (Z, SIGNAL, REFLECTIVITY, etc)")
        print("  P           - Cycle point size")
        print("  O           - Toggle rings/column poses")
        print("  B           - Toggle bounding boxes")
        print("  R           - Reset camera to birds-eye view")
        print("  ESC/Q       - Quit")
        print("\nMouse:")
        print("  Left drag   - Rotate view")
        print("  Right drag  - Pan")
        print("  Scroll      - Zoom")
        print(f"{'='*70}\n")
    
    def detection_to_cuboid_params(self, detection):
        """
        Convert 2D BEV detection to Ouster Cuboid parameters.
        
        BEV Coordinate System (matching MSAAI521 training):
        - Image: 640x640 pixels
        - X-axis (horizontal): -50m to +50m (left to right in image)
        - Y-axis (vertical): -50m to +50m (bottom to top in image, inverted from typical image coords)
        - Image center (320, 320) = world origin (0, 0)
        
        Returns:
            transform: 4x4 transformation matrix (includes scale)
            rgba: color tuple
        """
        x1, y1, x2, y2 = detection['box']
        
        # Box center and dimensions in pixels
        cx_px = (x1 + x2) / 2
        cy_px = (y1 + y2) / 2
        w_px = x2 - x1
        h_px = y2 - y1
        
        # BEV parameters (100m range, 1280 pixels)
        resolution = 0.078125  # meters per pixel (100m / 1280px)
        img_size = 1280
        x_range = (-50, 50)  # meters
        y_range = (-50, 50)  # meters
        
        # Convert pixel coordinates to world meters
        # X: left-to-right in image maps to X-axis in world
        x_m = x_range[0] + cx_px * resolution
        
        # Y: top-to-bottom in image is INVERTED from Y-axis in world
        # Image Y=0 (top) = +50m, Image Y=639 (bottom) = -50m
        y_m = y_range[1] - cy_px * resolution
        
        # Box dimensions in meters
        length_m = w_px * resolution
        width_m = h_px * resolution
        
        # Height based on class
        class_heights = {0: 1.5, 1: 2.5, 2: 1.7, 3: 1.7}
        height_m = class_heights.get(detection['class_id'], 1.5)
        
        # Z position - find ground from point cloud statistics
        # Assuming ground is around -1.8m (typical LiDAR mount height)
        center_z = -1.8 + height_m / 2
        
        # Create 4x4 transform matrix with scale
        # Cuboid is unit cube, scale it to actual dimensions
        transform = np.array([
            [length_m, 0, 0, x_m],
            [0, width_m, 0, y_m],
            [0, 0, height_m, center_z],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        color = self.class_colors.get(detection['class_id'], (1, 1, 1, 0.3))
        
        return transform, color
    

    
    def run(self, source_path, json_path=None):
        """Run visualization with Ouster SDK.
        
        Args:
            source_path: Path to PCAP or OSF file
            json_path: Path to JSON metadata (required for PCAP, optional for OSF)
        """
        source_path = Path(source_path)
        
        # Detect file format
        file_ext = source_path.suffix.lower()
        is_osf = file_ext == '.osf'
        is_pcap = file_ext == '.pcap'
        
        if not (is_osf or is_pcap):
            raise ValueError(f"Unsupported file format: {file_ext}. Use .pcap or .osf")
        
        # Load metadata based on file type
        if is_osf:
            print(f"Opening OSF file: {source_path}")
            print("  (metadata embedded in OSF)")
            source = open_source(str(source_path))
            # Get metadata from OSF - it returns a list, take first sensor
            metadata_list = source.metadata
            if isinstance(metadata_list, list):
                info = metadata_list[0]  # Use first sensor stream
                if len(metadata_list) > 1:
                    print(f"  Note: OSF contains {len(metadata_list)} sensor streams, using first")
            else:
                info = metadata_list
        else:  # PCAP
            if json_path is None:
                raise ValueError("PCAP files require --json parameter with sensor metadata")
            
            json_path = Path(json_path)
            print(f"Loading metadata from: {json_path}")
            with open(json_path, 'r') as f:
                metadata_str = f.read()
            
            info = SensorInfo(metadata_str)
            print(f"Opening PCAP: {source_path}")
            source = open_source(str(source_path), sensor_idx=0, meta=[str(json_path)])
        
        print(f"  Sensor: {info.prod_line}")
        print(f"  Resolution: {info.w}x{info.h}")
        print(f"  Format: {'OSF' if is_osf else 'PCAP'}")
        
        xyzlut = XYZLut(info)
        
        # Create Ouster PointViz
        point_viz = viz.PointViz("LiDAR Detection Viewer", window_width=1920, window_height=1080)
        
        # Add default keyboard/mouse controls
        viz.add_default_controls(point_viz)
        
        # Create cloud with maximum possible points
        max_points = info.format.pixels_per_column * info.format.columns_per_frame
        cloud = viz.Cloud(max_points)
        cloud.set_point_size(2)
        point_viz.add(cloud)
        
        # Storage for cuboids and labels - we'll add/remove dynamically
        cuboids = []
        labels = []
        max_cuboids = 50
        
        # Pre-create cuboid objects with transparent color (hidden)
        hidden_transform = np.eye(4, dtype=np.float32)
        for i in range(max_cuboids):
            cuboid = viz.Cuboid(hidden_transform, (0, 0, 0, 0))  # Fully transparent
            point_viz.add(cuboid)
            cuboids.append(cuboid)
            
            # Create label for each cuboid
            label = viz.Label("", 0, 0, 0)
            label.set_scale(2.0)  # Increased from 0.5 to 2.0 for better visibility
            point_viz.add(label)
            labels.append(label)
        
        # Key handler for controls
        def key_handler(ctx, key, mods):
            # Toggle boxes with 'B' key
            if key == ord('B') or key == ord('b'):
                self.show_boxes = not self.show_boxes
                print(f"\rBounding boxes: {'ON' if self.show_boxes else 'OFF'}    ", end='', flush=True)
                # Hide all boxes if turned off
                if not self.show_boxes:
                    for cuboid in cuboids:
                        cuboid.set_rgba((0, 0, 0, 0))
                    for label in labels:
                        label.set_text("")
                    point_viz.update()
                return True
            # Toggle pause with SPACE key
            elif key == 32:  # SPACE bar
                self.paused = not self.paused
                print(f"\r{'PAUSED' if self.paused else 'PLAYING'}    ", end='', flush=True)
                return True
            return False
        
        # Register key handler
        point_viz.push_key_handler(key_handler)
        
        # Make window visible and start rendering
        point_viz.running(True)
        point_viz.visible(True)
        point_viz.update()
        
        print(f"\n▶ Starting visualization...")
        print("Press SPACE to pause/play, . to step forward, ESC to quit\n")
        
        # Performance tracking
        timings = defaultdict(list)
        
        # Process frames
        try:
            frame_count = 0
            scan_iterator = iter(source)
            current_scan = None
            
            while True:
                # Check if window is still open
                if not point_viz.running():
                    print(f"\nWindow closed at frame {frame_count}")
                    break
                
                # Handle pause - keep rendering current frame but don't advance
                if self.paused:
                    point_viz.run_once()
                    time.sleep(0.05)  # Small delay to reduce CPU usage
                    continue
                
                # Get next scan only when not paused
                try:
                    scan_data = next(scan_iterator)
                except StopIteration:
                    print(f"\nReached end of PCAP file at frame {frame_count}")
                    break
                
                # Get scan
                t0 = time.time()
                scan = scan_data[0] if isinstance(scan_data, list) else scan_data
                if scan is None:
                    continue
                
                # Convert to XYZ
                t1 = time.time()
                xyz = xyzlut(scan)
                points = xyz.reshape(-1, 3)
                timings['xyz_convert'].append(time.time() - t1)
                
                # Use all points (including zeros) to maintain size
                t1 = time.time()
                cloud.set_xyz(points)
                timings['set_xyz'].append(time.time() - t1)
                
                # Color by reflectivity (like ouster-cli) - normalized to 0-1 range
                t1 = time.time()
                reflectivity = scan.field(ChanField.REFLECTIVITY).reshape(-1).astype(np.float32)
                # Normalize reflectivity for better visualization
                refl_min, refl_max = np.percentile(reflectivity[reflectivity > 0], [5, 95])
                refl_normalized = np.clip((reflectivity - refl_min) / (refl_max - refl_min + 1e-6), 0, 1)
                cloud.set_key(refl_normalized)
                timings['set_color'].append(time.time() - t1)
                
                # Run detection every N frames (synchronous)
                if self.show_boxes and (frame_count % self.detection_skip == 0):
                    # Prepare point cloud for detection
                    t1 = time.time()
                    reflectivity = scan.field(ChanField.REFLECTIVITY).reshape(-1)
                    valid_mask = ~np.all(points == 0, axis=1)
                    valid_points = points[valid_mask]
                    valid_reflectivity = reflectivity[valid_mask]
                    points_full = np.column_stack([valid_points, valid_reflectivity.astype(np.float32)])
                    timings['prep_detection'].append(time.time() - t1)
                    
                    # Run detection on THIS frame
                    t1 = time.time()
                    bev_image = self.inference.preprocess_lidar(points_full)
                    timings['bev_gen'].append(time.time() - t1)
                    
                    t1 = time.time()
                    yolo_results = self.inference.predict(bev_image)
                    timings['yolo_inference'].append(time.time() - t1)
                    
                    # Parse detections
                    t1 = time.time()
                    self.latest_detections = []
                    if yolo_results.boxes is not None:
                        for box in yolo_results.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            self.latest_detections.append({
                                'box': [x1, y1, x2, y2],
                                'class_id': cls,
                                'class_name': self.inference.class_names.get(cls, f'class_{cls}'),
                                'confidence': conf
                            })
                    timings['parse_detections'].append(time.time() - t1)
                
                # Update cuboids and labels with detections from THIS frame
                if self.show_boxes:
                    # Update cuboids and labels with latest detections
                    t1 = time.time()
                    num_detections = len(self.latest_detections)
                    for i, cuboid in enumerate(cuboids):
                        if i < num_detections:
                            det = self.latest_detections[i]
                            transform, color = self.detection_to_cuboid_params(det)
                            
                            cuboid.set_transform(transform)
                            cuboid.set_rgba(color)
                            
                            # Update label with class name and confidence
                            label = labels[i]
                            label_text = f"{det['class_name']} {det['confidence']:.2f}"
                            label.set_text(label_text)
                            # Position label at top of bounding box
                            x1, y1, x2, y2 = det['box']
                            cx_px = (x1 + x2) / 2
                            cy_px = (y1 + y2) / 2
                            resolution = self.inference.rasterizer.resolution
                            cx_m = (cx_px - 640) * resolution
                            cy_m = (640 - cy_px) * resolution
                            label.set_position(cx_m, cy_m, 3.0)  # 3m high
                        else:
                            # Hide unused cuboids and labels
                            cuboid.set_rgba((0, 0, 0, 0))
                            labels[i].set_text("")
                    timings['update_cuboids'].append(time.time() - t1)
                
                # Update visualization and process events
                t1 = time.time()
                point_viz.update()
                timings['viz_update'].append(time.time() - t1)
                
                t1 = time.time()
                point_viz.run_once()  # Process UI events non-blocking
                timings['viz_render'].append(time.time() - t1)
                
                timings['total_frame'].append(time.time() - t0)
                
                # Limit to ~10 FPS for better viewing and keeping up with detection
                frame_time = time.time() - t0
                target_frame_time = 1.0 / 10.0  # 10 FPS
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
                
                frame_count += 1
                self.current_frame = frame_count
                
                if frame_count % 30 == 0:
                    # Print performance stats every 30 frames
                    avg_times = {k: np.mean(v[-30:]) * 1000 for k, v in timings.items() if v}
                    fps = 1.0 / (np.mean(timings['total_frame'][-30:]) + 1e-6)
                    print(f"\n{'='*70}")
                    print(f"Frame {frame_count} | FPS: {fps:.1f} | Detections: {len(self.latest_detections)}")
                    print(f"{'='*70}")
                    for key in ['xyz_convert', 'set_xyz', 'set_color', 'bev_gen', 'yolo_inference', 
                               'parse_detections', 'update_cuboids', 'viz_update', 'viz_render', 'total_frame']:
                        if key in avg_times:
                            print(f"  {key:20s}: {avg_times[key]:6.2f} ms")
                    print(f"{'='*70}\n")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n✗ Error during visualization: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print(f"\n{'='*70}")
            print("Session Summary")
            print(f"{'='*70}")
            print(f"  Frames processed: {self.current_frame}")
            print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Ouster SDK Visualization with Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PCAP file with JSON metadata
  python pcap_visualizer_ouster.py --source data.pcap --json metadata.json --model model.pt
  
  # OSF file (metadata embedded)
  python pcap_visualizer_ouster.py --source data.osf --model model.pt
  
  # Legacy PCAP argument (deprecated, use --source)
  python pcap_visualizer_ouster.py --pcap data.pcap --json metadata.json --model model.pt
        """
    )
    
    parser.add_argument('--source', type=str, help='Path to PCAP or OSF file')
    parser.add_argument('--pcap', type=str, help='Path to PCAP file (legacy, use --source instead)')
    parser.add_argument('--json', type=str, help='Path to sensor JSON metadata (required for PCAP)')
    parser.add_argument('--model', type=str, required=True, help='Path to model (.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--skip', type=int, default=1, help='Run detection every N frames (1=every frame, 3=every 3rd frame)')
    
    args = parser.parse_args()
    
    # Handle legacy --pcap argument
    if args.pcap and not args.source:
        print("Warning: --pcap is deprecated, use --source instead")
        args.source = args.pcap
    
    if not args.source:
        parser.error("--source is required (path to PCAP or OSF file)")
    
    # Validate source file exists
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"✗ Source file not found: {args.source}")
        return 1
    
    # Check if JSON is required (for PCAP)
    if source_path.suffix.lower() == '.pcap' and not args.json:
        print(f"✗ PCAP files require --json parameter with sensor metadata")
        return 1
    
    # Validate JSON if provided
    if args.json and not Path(args.json).exists():
        print(f"✗ JSON metadata not found: {args.json}")
        return 1
    
    # Validate model
    if not Path(args.model).exists():
        print(f"✗ Model not found: {args.model}")
        return 1
    
    visualizer = OusterDetectionViz(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        detection_skip=args.skip
    )
    
    visualizer.run(args.source, args.json)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
