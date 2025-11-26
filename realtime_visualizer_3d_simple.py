"""
Simple Real-time 3D LiDAR Object Detection Visualizer

Similar to the 2D visualizer but shows point cloud in 3D with matplotlib.
Simple, clean interface with frame-by-frame playback.

Usage:
    python realtime_visualizer_3d_simple.py \
        --source build/data/raw/v1.0-trainval/samples/LIDAR_TOP \
        --model build/runs/detect/stage2_finetune/weights/best.pt
"""

import numpy as np
from pathlib import Path
import argparse
import time
from collections import deque
import sys

import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.append(str(Path(__file__).parent))
from src.Inference.LidarInference import LidarInference


class Simple3DVisualizer:
    """
    Simple real-time 3D LiDAR object detection visualizer.
    """
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the visualizer.
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        # Initialize inference engine
        self.inference = LidarInference(model_path, conf_threshold, iou_threshold)
        
        # Visualization settings
        self.window_name = "Real-time 3D LiDAR Object Detection"
        
        # Playback controls
        self.paused = False
        self.frame_delay = 0.05  # seconds between frames
        self.current_frame = 0
        self.total_frames = 0
        self.quit_flag = False
        
        # Statistics
        self.fps_buffer = deque(maxlen=30)
        
        # View control
        self.zoom_level = 1.0
        self.base_xlim = [-50, 50]
        self.base_ylim = [-50, 50]
        self.base_zlim = [-2, 10]
        
        # Class colors
        self.class_colors = {
            0: 'red',      # car
            1: 'blue',     # truck_bus
            2: 'green',    # pedestrian
            3: 'yellow',   # cyclist
        }
        
        print(f"\n{'='*70}")
        print("Simple 3D LiDAR Object Detection Visualizer")
        print(f"{'='*70}")
        print("\nControls:")
        print("  SPACE       - Pause/Resume")
        print("  Q           - Quit")
        print("  +/-         - Increase/Decrease speed")
        print("  N           - Next frame (when paused)")
        print("  R           - Reset to first frame")
        print("  Mouse Scroll - Zoom in/out")
        print("  Mouse Drag   - Rotate view")
        print(f"\n{'='*70}\n")
    
    def load_lidar_source(self, source_path):
        """Load LiDAR files from directory."""
        source_path = Path(source_path)
        
        if source_path.is_dir():
            lidar_files = sorted(list(source_path.glob('*.pcd.bin')))
            self.total_frames = len(lidar_files)
            print(f"✓ Loaded {self.total_frames} LiDAR frames")
            return lidar_files
        else:
            print(f"✗ Source must be a directory: {source_path}")
            return []
    
    def create_3d_box(self, center, size, rotation=0):
        """
        Create 3D bounding box vertices.
        
        Args:
            center: [x, y, z] center of box
            size: [length, width, height]
            rotation: rotation around z-axis (radians)
        
        Returns:
            vertices: 8x3 array of box corners
            edges: list of edge indices
        """
        l, w, h = size
        x, y, z = center
        
        # 8 corners of box (before rotation)
        corners = np.array([
            [-l/2, -w/2, -h/2],
            [ l/2, -w/2, -h/2],
            [ l/2,  w/2, -h/2],
            [-l/2,  w/2, -h/2],
            [-l/2, -w/2,  h/2],
            [ l/2, -w/2,  h/2],
            [ l/2,  w/2,  h/2],
            [-l/2,  w/2,  h/2],
        ])
        
        # Rotation matrix around z-axis
        if rotation != 0:
            cos_r, sin_r = np.cos(rotation), np.sin(rotation)
            rot_mat = np.array([
                [cos_r, -sin_r, 0],
                [sin_r,  cos_r, 0],
                [0, 0, 1]
            ])
            corners = corners @ rot_mat.T
        
        # Translate to center
        corners += center
        
        # Define edges (pairs of vertex indices)
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
        ]
        
        return corners, edges
    
    def detection_to_3d_box(self, detection):
        """Convert YOLO detection to 3D box parameters."""
        # Get BEV box in pixels (xyxy format)
        x1, y1, x2, y2 = detection['box']
        
        # Calculate center and size in pixels
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        w = x2 - x1  # width in pixels (horizontal in image)
        h = y2 - y1  # height in pixels (vertical in image)
        
        # Convert to meters using BEVRasterizer mapping:
        # Image X (columns/width) -> World X (forward)
        # Image Y (rows/height, flipped) -> World Y (left)
        resolution = self.inference.rasterizer.resolution
        x_min = self.inference.rasterizer.x_range[0]
        y_min = self.inference.rasterizer.y_range[0]
        img_height = self.inference.rasterizer.height
        
        # Convert pixel coordinates to world coordinates (matching BEVRasterizer)
        x_m = x_min + x_center * resolution
        y_m = y_min + (img_height - 1 - y_center) * resolution
        
        # Box dimensions (no swap needed - already aligned)
        length_m = w * resolution   # Image width -> World X (forward/length)
        width_m = h * resolution    # Image height -> World Y (left/width)
        
        # Default heights by class
        class_heights = {0: 1.5, 1: 2.5, 2: 1.7, 3: 1.7}
        height_m = class_heights.get(detection['class_id'], 1.5)
        
        # Place box on ground (z=0 at bottom of box, not center)
        center = np.array([x_m, y_m, 0])
        size = np.array([length_m, width_m, height_m])
        
        return center, size, 0  # No rotation needed
    
    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == ' ':
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "PLAYING"
            print(f"\n{status}")
        elif event.key == 'q':
            self.quit_flag = True
            plt.close('all')
        elif event.key == '+' or event.key == '=':
            self.frame_delay = max(0.001, self.frame_delay * 0.8)
            print(f"Speed up: {1/self.frame_delay:.1f} FPS")
        elif event.key == '-':
            self.frame_delay = min(1.0, self.frame_delay * 1.25)
            print(f"Speed down: {1/self.frame_delay:.1f} FPS")
        elif event.key == 'n' and self.paused:
            self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
        elif event.key == 'r':
            self.current_frame = 0
            print("Reset to frame 0")
    
    def on_scroll(self, event):
        """Handle mouse scroll for zoom."""
        # Update zoom level
        if event.button == 'up':
            self.zoom_level *= 0.9  # Zoom in
        elif event.button == 'down':
            self.zoom_level *= 1.1  # Zoom out
        
        # Clamp zoom level
        self.zoom_level = max(0.1, min(5.0, self.zoom_level))
        
        # Force redraw with new zoom
        if hasattr(self, 'current_ax') and self.current_ax is not None:
            self.update_axis_limits(self.current_ax)
            plt.draw()
    
    def update_axis_limits(self, ax):
        """Update axis limits based on zoom level."""
        ax.set_xlim([self.base_xlim[0] * self.zoom_level, self.base_xlim[1] * self.zoom_level])
        ax.set_ylim([self.base_ylim[0] * self.zoom_level, self.base_ylim[1] * self.zoom_level])
        ax.set_zlim([self.base_zlim[0] * self.zoom_level, self.base_zlim[1] * self.zoom_level])
    
    def visualize_frame(self, ax, points, detections, frame_info):
        """
        Visualize a single frame in 3D.
        
        Args:
            ax: matplotlib 3D axis
            points: Nx4 array of point cloud [x, y, z, intensity]
            detections: List of detection dictionaries
            frame_info: Frame metadata
        """
        ax.clear()
        
        # Subsample points for performance
        subsample = max(1, len(points) // 10000)
        points_sub = points[::subsample]
        
        # Plot point cloud in blue (matching 2D visualizer style)
        ax.scatter(
            points_sub[:, 0],
            points_sub[:, 1],
            points_sub[:, 2],
            c='cyan',
            s=1,
            alpha=0.4,
            marker='.'
        )
        
        # Plot bounding boxes
        for det in detections:
            center, size, rotation = self.detection_to_3d_box(det)
            corners, edges = self.create_3d_box(center, size, rotation)
            
            color = self.class_colors.get(det['class_id'], 'white')
            
            # Draw edges
            for edge in edges:
                points_edge = corners[edge]
                ax.plot3D(
                    points_edge[:, 0],
                    points_edge[:, 1],
                    points_edge[:, 2],
                    color=color,
                    linewidth=2,
                    alpha=0.8
                )
        
        # Set view to bird's eye
        ax.view_init(elev=70, azim=-90)
        
        # Set axis properties with current zoom level
        self.update_axis_limits(ax)
        ax.set_xlabel('X (forward) [m]', fontsize=8)
        ax.set_ylabel('Y (left) [m]', fontsize=8)
        ax.set_zlabel('Z (up) [m]', fontsize=8)
        ax.set_box_aspect([2, 2, 0.3])
        
        # Turn off grid and set dark background
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # Draw distance rings on the ground plane
        theta = np.linspace(0, 2*np.pi, 100)
        for radius in [10, 20, 30, 40]:
            x_ring = radius * np.cos(theta)
            y_ring = radius * np.sin(theta)
            z_ring = np.zeros_like(theta)
            ax.plot3D(x_ring, y_ring, z_ring, color='gray', linewidth=1, alpha=0.5)
        
        # Title with stats
        fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
        status = "PAUSED" if self.paused else "PLAYING"
        
        title = f"Frame: {frame_info['frame_num']}/{frame_info['total_frames']}"
        title += f" | FPS: {fps:.1f}"
        title += f" | Detections: {len(detections)}"
        title += f"\n{status}"
        
        ax.set_title(title, fontsize=10, pad=10)
        
        # Count by class
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Add legend
        legend_text = []
        for class_name, count in sorted(class_counts.items()):
            legend_text.append(f"{class_name}: {count}")
        
        if legend_text:
            ax.text2D(
                0.98, 0.02, '\n'.join(legend_text),
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
                color='white'
            )
    
    def run(self, lidar_files):
        """
        Run the visualization loop.
        
        Args:
            lidar_files: List of LiDAR file paths
        """
        if not lidar_files:
            print("✗ No LiDAR files to process")
            return
        
        # Create figure and axis with dark background
        fig = plt.figure(figsize=(12, 8), facecolor='#000033')
        ax = fig.add_subplot(111, projection='3d', facecolor='#000033')
        fig.canvas.manager.set_window_title(self.window_name)
        
        # Store axis reference for zoom
        self.current_ax = ax
        
        # Connect keyboard and mouse handlers
        fig.canvas.mpl_connect('key_press_event', self.on_key)
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Make it non-blocking
        plt.ion()
        plt.show()
        
        print(f"\nStarting visualization...")
        print(f"Total frames: {self.total_frames}")
        print(f"Press SPACE to pause, Q to quit\n")
        
        # Main loop
        try:
            while self.current_frame < self.total_frames and not self.quit_flag:
                loop_start = time.time()
                
                if not self.paused:
                    # Load and process frame
                    lidar_file = lidar_files[self.current_frame]
                    
                    # Load points
                    points = self.inference.load_lidar_from_file(lidar_file)
                    
                    # Run inference
                    results = self.inference.predict_from_lidar_file(lidar_file)
                    detections = results['detections']
                    
                    # Calculate FPS
                    processing_time = time.time() - loop_start
                    fps = 1.0 / processing_time if processing_time > 0 else 0
                    self.fps_buffer.append(fps)
                    
                    # Create visualization
                    frame_info = {
                        'frame_num': self.current_frame + 1,
                        'total_frames': self.total_frames,
                        'filename': lidar_file.name
                    }
                    
                    self.visualize_frame(ax, points, detections, frame_info)
                    
                    # Update display
                    plt.draw()
                    plt.pause(0.001)
                    
                    # Move to next frame
                    self.current_frame += 1
                    
                    # Frame rate control
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, self.frame_delay - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    # Paused - just wait and check for input
                    plt.pause(0.1)
                
                # Check if window was closed
                if not plt.fignum_exists(fig.number):
                    break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            plt.close('all')
            print("\nVisualization ended")
            print(f"Processed {self.current_frame} frames")


def main():
    parser = argparse.ArgumentParser(description="Simple 3D LiDAR Object Detection Visualizer")
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to LiDAR data directory'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained YOLO model (.pt file)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections (default: 0.25)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process'
    )
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = Simple3DVisualizer(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Load data source
    lidar_files = visualizer.load_lidar_source(args.source)
    
    if not lidar_files:
        print("✗ No LiDAR files found")
        return
    
    # Limit frames if requested
    if args.max_frames:
        lidar_files = lidar_files[:args.max_frames]
        visualizer.total_frames = len(lidar_files)
        print(f"✓ Limited to first {args.max_frames} frames")
    
    # Run visualization
    visualizer.run(lidar_files)


if __name__ == '__main__':
    main()
