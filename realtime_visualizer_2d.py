"""
Real-time LiDAR Object Detection Visualizer

This script provides a real-time visualization interface for LiDAR-based object detection.
It can process:
1. PCAP files (Velodyne/Ouster LiDAR)
2. ROS bag files
3. Directory of LiDAR files
4. Live LiDAR stream

Features:
- Frame-by-frame playback with speed control
- Real-time bounding box visualization
- Detection statistics overlay
- Bird's Eye View (BEV) display
- Pause/resume, frame stepping
- Export detections to video or CSV

Usage:
    # From directory of LiDAR files
    python realtime_visualizer.py --source build/data/raw/v1.0-trainval/samples/LIDAR_TOP \
                                   --model build/runs/detect/stage2_finetune/weights/best.pt
    
    # From PCAP file (requires pcap parser)
    python realtime_visualizer.py --source data.pcap --model model.pt --format pcap
    
    # With recording
    python realtime_visualizer.py --source lidar_dir/ --model model.pt --record output.mp4
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from collections import deque
import sys

sys.path.append(str(Path(__file__).parent))
from src.Inference.LidarInference import LidarInference


class RealtimeVisualizer:
    """
    Real-time LiDAR object detection visualizer with interactive controls.
    """
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the real-time visualizer.
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        # Initialize inference engine
        self.inference = LidarInference(model_path, conf_threshold, iou_threshold)
        
        # Visualization settings
        self.window_name = "Real-time LiDAR Object Detection"
        self.display_width = 1280
        self.display_height = 1280
        
        # Playback controls
        self.paused = False
        self.frame_delay = 50  # milliseconds between frames (default: 20 FPS)
        self.current_frame = 0
        self.total_frames = 0
        
        # Statistics tracking
        self.fps_buffer = deque(maxlen=30)
        self.detection_history = deque(maxlen=100)
        
        # Recording
        self.video_writer = None
        self.recording = False
        
        print(f"\n{'='*70}")
        print("Real-time LiDAR Object Detection Visualizer")
        print(f"{'='*70}")
        print("\nControls:")
        print("  SPACE   - Pause/Resume")
        print("  Q/ESC   - Quit")
        print("  +/-     - Increase/Decrease speed")
        print("  N       - Next frame (when paused)")
        print("  R       - Reset to first frame")
        print("  S       - Save current frame")
        print(f"\n{'='*70}\n")
    
    def load_lidar_source(self, source_path, source_format='auto'):
        """
        Load LiDAR data from various sources.
        
        Args:
            source_path: Path to data source (directory, PCAP, ROS bag)
            source_format: Format type ('auto', 'directory', 'pcap', 'rosbag')
        
        Returns:
            List of LiDAR file paths or data objects
        """
        source_path = Path(source_path)
        
        if source_format == 'auto':
            if source_path.is_dir():
                source_format = 'directory'
            elif source_path.suffix == '.pcap':
                source_format = 'pcap'
            elif source_path.suffix == '.bag':
                source_format = 'rosbag'
        
        if source_format == 'directory':
            # Load all .pcd.bin files from directory
            lidar_files = sorted(list(source_path.glob('*.pcd.bin')))
            print(f"✓ Loaded {len(lidar_files)} LiDAR frames from directory")
            return lidar_files
        
        elif source_format == 'pcap':
            print(f"✗ PCAP parsing not implemented yet")
            print(f"  Install: pip install python-pcapng")
            print(f"  For now, use directory of .pcd.bin files")
            return []
        
        elif source_format == 'rosbag':
            print(f"✗ ROS bag parsing not implemented yet")
            print(f"  Install: pip install rosbag")
            print(f"  For now, use directory of .pcd.bin files")
            return []
        
        return []
    
    def create_visualization(self, bev_image, detections, frame_info):
        """
        Create visualization with BEV image, bounding boxes, and overlays.
        
        Args:
            bev_image: BEV image (H, W, 3)
            detections: List of detection dictionaries
            frame_info: Dictionary with frame metadata
        
        Returns:
            Visualization image ready for display
        """
        # Start with BEV image
        vis = bev_image.copy()
        
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            cls = det['class_id']
            conf = det['confidence']
            
            # Get color
            color = self.inference.class_colors.get(cls, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class_name']}: {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            cv2.putText(
                vis, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        
        # Add overlays
        self._draw_overlays(vis, detections, frame_info)
        
        return vis
    
    def _draw_overlays(self, image, detections, frame_info):
        """
        Draw information overlays on the visualization.
        
        Args:
            image: Image to draw on (modified in-place)
            detections: List of detections
            frame_info: Frame metadata
        """
        h, w = image.shape[:2]
        
        # Semi-transparent overlay for text background
        overlay = image.copy()
        
        # Top-left: Frame info
        info_lines = [
            f"Frame: {frame_info['frame_num']}/{frame_info['total_frames']}",
            f"FPS: {frame_info['fps']:.1f}",
            f"Detections: {len(detections)}",
            f"{'PAUSED' if self.paused else 'PLAYING'}"
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.rectangle(overlay, (5, y_offset - 20), (250, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(image, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        # Apply transparency
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Top-right: Detection breakdown
        if detections:
            class_counts = {}
            for det in detections:
                cls = det['class_name']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            y_offset = 30
            for cls, count in sorted(class_counts.items()):
                text = f"{cls}: {count}"
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(overlay, (w - text_width - 20, y_offset - 20),
                            (w - 5, y_offset + 5), (0, 0, 0), -1)
                cv2.putText(image, text, (w - text_width - 15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
            
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Bottom: Controls hint
        controls = "SPACE: Pause | +/-: Speed | N: Next | S: Save | Q: Quit"
        (text_width, text_height), _ = cv2.getTextSize(
            controls, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(overlay, (5, h - 35), (w - 5, h - 5), (0, 0, 0), -1)
        cv2.putText(image, controls, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Center: Vehicle indicator
        center_x, center_y = w // 2, h // 2
        cv2.circle(image, (center_x, center_y), 15, (0, 255, 255), -1)
        cv2.putText(image, "EGO", (center_x - 20, center_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Distance circles (20m, 40m)
        for radius_m in [20, 40]:
            radius_px = int(radius_m / self.inference.rasterizer.resolution)
            cv2.circle(image, (center_x, center_y), radius_px, (100, 100, 100), 1)
    
    def setup_recording(self, output_path):
        """
        Setup video recording.
        
        Args:
            output_path: Path to save output video
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            1000.0 / self.frame_delay,  # FPS
            (self.display_width, self.display_height)
        )
        self.recording = True
        print(f"✓ Recording to: {output_path}")
    
    def process_frame(self, lidar_file):
        """
        Process a single LiDAR frame.
        
        Args:
            lidar_file: Path to LiDAR file
        
        Returns:
            results: Detection results dictionary
        """
        start_time = time.time()
        
        # Run inference
        results = self.inference.predict_from_lidar_file(lidar_file)
        
        # Calculate FPS
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        self.fps_buffer.append(fps)
        
        return results
    
    def run(self, lidar_files, record_path=None):
        """
        Run the real-time visualization loop.
        
        Args:
            lidar_files: List of LiDAR file paths
            record_path: Optional path to save video recording
        """
        if not lidar_files:
            print("✗ No LiDAR files to process")
            return
        
        self.total_frames = len(lidar_files)
        
        # Setup recording if requested
        if record_path:
            self.setup_recording(record_path)
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        
        # Main loop
        while self.current_frame < self.total_frames:
            loop_start = time.time()
            
            if not self.paused:
                # Process current frame
                lidar_file = lidar_files[self.current_frame]
                results = self.process_frame(lidar_file)
                
                # Create visualization
                frame_info = {
                    'frame_num': self.current_frame + 1,
                    'total_frames': self.total_frames,
                    'fps': np.mean(self.fps_buffer) if self.fps_buffer else 0,
                    'filename': lidar_file.name
                }
                
                vis_image = self.create_visualization(
                    results['bev_image'],
                    results['detections'],
                    frame_info
                )
                
                # Display
                cv2.imshow(self.window_name, vis_image)
                
                # Record if enabled
                if self.recording and self.video_writer:
                    self.video_writer.write(vis_image)
                
                # Store for statistics
                self.detection_history.append(len(results['detections']))
                
                # Move to next frame
                self.current_frame += 1
            
            # Handle keyboard input
            elapsed = (time.time() - loop_start) * 1000  # to milliseconds
            wait_time = max(1, int(self.frame_delay - elapsed))
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("\nQuitting...")
                break
            elif key == ord(' '):  # SPACE
                self.paused = not self.paused
                print(f"{'Paused' if self.paused else 'Resumed'}")
            elif key == ord('+') or key == ord('='):
                self.frame_delay = max(10, self.frame_delay - 10)
                print(f"Speed increased (delay: {self.frame_delay}ms)")
            elif key == ord('-') or key == ord('_'):
                self.frame_delay = min(500, self.frame_delay + 10)
                print(f"Speed decreased (delay: {self.frame_delay}ms)")
            elif key == ord('n') and self.paused:  # Next frame
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
            elif key == ord('r'):  # Reset
                self.current_frame = 0
                print("Reset to first frame")
            elif key == ord('s'):  # Save frame
                save_path = f"frame_{self.current_frame:05d}.png"
                cv2.imwrite(save_path, vis_image)
                print(f"Saved frame to: {save_path}")
        
        # Cleanup
        self.cleanup()
        
        # Print statistics
        self.print_statistics()
    
    def cleanup(self):
        """Clean up resources."""
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
    
    def print_statistics(self):
        """Print session statistics."""
        print(f"\n{'='*70}")
        print("Session Statistics")
        print(f"{'='*70}")
        print(f"Total frames processed: {self.current_frame}")
        print(f"Average FPS: {np.mean(self.fps_buffer):.1f}")
        if self.detection_history:
            print(f"Average detections per frame: {np.mean(self.detection_history):.1f}")
            print(f"Max detections in a frame: {max(self.detection_history)}")
        if self.recording:
            print(f"✓ Video saved")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time LiDAR Object Detection Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process directory of LiDAR files
  python realtime_visualizer.py --source build/data/raw/v1.0-trainval/samples/LIDAR_TOP \\
                                 --model build/runs/detect/stage2_finetune/weights/best.pt
  
  # With recording
  python realtime_visualizer.py --source lidar_dir/ --model model.pt --record output.mp4
  
  # Lower confidence threshold for more detections
  python realtime_visualizer.py --source lidar_dir/ --model model.pt --conf 0.15
        """
    )
    
    parser.add_argument('--source', type=str, required=True,
                       help='Path to LiDAR data (directory, PCAP, or ROS bag)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--format', type=str, default='auto',
                       choices=['auto', 'directory', 'pcap', 'rosbag'],
                       help='Input format (auto-detect by default)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--record', type=str, default=None,
                       help='Save visualization to video file (e.g., output.mp4)')
    parser.add_argument('--fps', type=int, default=20,
                       help='Playback FPS (default: 20)')
    
    args = parser.parse_args()
    
    # Verify model exists
    if not Path(args.model).exists():
        print(f"✗ Model not found: {args.model}")
        return 1
    
    # Initialize visualizer
    visualizer = RealtimeVisualizer(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Set playback speed
    visualizer.frame_delay = int(1000 / args.fps)
    
    # Load LiDAR source
    lidar_files = visualizer.load_lidar_source(args.source, args.format)
    
    if not lidar_files:
        print(f"✗ No LiDAR data found in: {args.source}")
        return 1
    
    # Run visualization
    try:
        visualizer.run(lidar_files, args.record)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        visualizer.cleanup()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        visualizer.cleanup()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
