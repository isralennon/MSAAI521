"""
RawDataInspector: A comprehensive tool for exploring and visualizing nuScenes autonomous driving dataset.

This class provides methods to inspect, analyze, and visualize various components of the nuScenes dataset
including LiDAR point clouds, 3D bounding box annotations, camera images, and scene metadata.

Technical Overview:
- Interfaces with nuScenes API to access dataset metadata and sensor data
- Processes LiDAR point clouds stored in binary .pcd.bin format
- Handles 3D coordinate transformations and spatial data visualization
- Generates matplotlib-based visualizations of multi-sensor autonomous vehicle data
"""

import numpy as np
import os
from pathlib import Path
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class RawDataInspector:
    """
    Inspector class for analyzing raw nuScenes dataset components.
    
    This class provides a suite of methods to explore autonomous vehicle sensor data,
    including LiDAR point clouds, camera images, and 3D object annotations. All visualizations
    are saved to disk rather than displayed interactively for compatibility with headless environments.
    
    Attributes:
        nusc: NuScenes instance providing access to the dataset API
        output_dir: Path object pointing to directory where visualizations are saved
    """
    
    def __init__(self, nusc, output_dir='build/visualizations'):
        """
        Initialize the RawDataInspector with a nuScenes dataset instance.
        
        Args:
            nusc: A NuScenes instance that provides API access to the dataset.
                  This object contains all the metadata tables (scenes, samples, annotations, etc.)
                  and methods to query and render the dataset.
            output_dir: String path where visualization files will be saved.
                       Defaults to 'build/visualizations'. Directory is created if it doesn't exist.
        
        Technical Details:
            - Creates output directory structure using pathlib for cross-platform compatibility
            - Stores nuScenes instance for accessing dataset metadata tables and file paths
        """
        self.nusc = nusc
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

    def inspect_point_cloud(self, sample_data_token):
        """
        Extract and analyze statistics from a LiDAR point cloud.
        
        This method loads a LiDAR point cloud file and computes spatial and intensity statistics
        to understand the point cloud's coverage and characteristics.
        
        Args:
            sample_data_token: String UUID identifying a specific LiDAR sensor capture in the dataset.
                              This token references an entry in the sample_data table.
        
        Returns:
            Dictionary containing:
                - shape: Tuple (4, N) where N is number of points. First 3 rows are x,y,z coordinates,
                        4th row is intensity values
                - num_points: Total number of 3D points captured by the LiDAR
                - x_range: Tuple (min, max) of x-coordinates in meters (forward/backward)
                - y_range: Tuple (min, max) of y-coordinates in meters (left/right)
                - z_range: Tuple (min, max) of z-coordinates in meters (up/down)
                - intensity_range: Tuple (min, max) of LiDAR return intensity values
        
        Technical Details:
            1. Queries nuScenes metadata to get file path for the LiDAR scan
            2. Loads binary point cloud file (.pcd.bin format) using nuScenes utilities
            3. Point cloud data structure: 4xN numpy array where:
               - Row 0: X coordinates (forward direction in vehicle frame)
               - Row 1: Y coordinates (left direction in vehicle frame)
               - Row 2: Z coordinates (up direction in vehicle frame)
               - Row 3: Intensity values (reflectivity of laser return)
            4. Computes min/max statistics along each dimension using numpy operations
        """
        # Retrieve metadata record for this LiDAR capture from sample_data table
        sample_data = self.nusc.get('sample_data', sample_data_token)
        
        # Construct absolute file path to the binary point cloud file
        pcl_path = os.path.join(self.nusc.dataroot, sample_data['filename'])

        # Load point cloud from binary file into LidarPointCloud object
        # File format: binary float32 array with 4 values per point (x, y, z, intensity)
        pc = LidarPointCloud.from_file(pcl_path)
        points = pc.points  # Access underlying numpy array (4, N)

        # Compute and return statistical summary of point cloud
        return {
            'shape': points.shape,  # (4, N) - 4 channels, N points
            'num_points': points.shape[1],  # Total number of points
            'x_range': (points[0].min(), points[0].max()),  # Forward/backward extent (meters)
            'y_range': (points[1].min(), points[1].max()),  # Left/right extent (meters)
            'z_range': (points[2].min(), points[2].max()),  # Up/down extent (meters)
            'intensity_range': (points[3].min(), points[3].max())  # LiDAR intensity values
        }

    def inspect_annotations(self, sample_token):
        """
        Extract all 3D bounding box annotations for a given sample (timestamp).
        
        Each sample in nuScenes represents a synchronized snapshot from all sensors at a specific
        timestamp. This method retrieves all object annotations (3D bounding boxes) associated
        with that sample.
        
        Args:
            sample_token: String UUID identifying a sample (multi-sensor snapshot at one timestamp).
                         References an entry in the sample table.
        
        Returns:
            List of dictionaries, one per annotated object, each containing:
                - category: String object class (e.g., 'vehicle.car', 'human.pedestrian.adult')
                - translation: [x, y, z] center of 3D bounding box in global coordinates (meters)
                - size: [width, length, height] dimensions of bounding box (meters)
                - rotation: Quaternion [w, x, y, z] representing 3D orientation of the box
        
        Technical Details:
            1. Queries sample table to get list of annotation tokens for this timestamp
            2. Each sample contains 'anns' field: list of annotation token UUIDs
            3. For each annotation token, queries sample_annotation table for full metadata
            4. Coordinate system: Global world coordinates (not ego vehicle frame)
            5. Rotation format: Quaternion for 3D rotation (avoids gimbal lock issues)
            6. Size convention: [width (left-right), length (forward-back), height (up-down)]
        """
        # Retrieve the sample record containing list of annotation tokens
        sample = self.nusc.get('sample', sample_token)

        # Collect detailed annotation data for each object in this sample
        annotations = []
        for ann_token in sample['anns']:  # Iterate over all annotation UUIDs
            # Query sample_annotation table for full annotation metadata
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Extract key 3D bounding box parameters
            annotations.append({
                'category': ann['category_name'],  # Object class label
                'translation': ann['translation'],  # 3D position: [x, y, z] in global frame (meters)
                'size': ann['size'],  # Box dimensions: [width, length, height] (meters)
                'rotation': ann['rotation']  # Orientation: quaternion [w, x, y, z]
            })

        return annotations

    def visualize_3d_scene(self, sample_token):
        """
        Create a 3D visualization of LiDAR point cloud with overlaid 3D bounding box annotations.
        
        This method generates a 3D scatter plot showing the spatial distribution of LiDAR points
        colored by height (z-coordinate), with red wireframe boxes representing annotated objects.
        The visualization is saved as a PNG file for inspection.
        
        Args:
            sample_token: String UUID identifying the sample (timestamp) to visualize.
        
        Technical Details:
            1. Data Loading:
               - Retrieves LIDAR_TOP sensor data token from the sample
               - Loads binary point cloud file and extracts xyz coordinates (discards intensity)
            
            2. Point Subsampling:
               - Randomly samples up to 10,000 points for performance (full clouds have ~30K points)
               - Uses numpy.random.choice with replace=False for uniform random sampling
               - Reduces rendering time while maintaining spatial distribution
            
            3. 3D Scatter Plot:
               - Creates matplotlib 3D axis using projection='3d'
               - Points colored by z-coordinate (height) using 'viridis' colormap
               - Small point size (s=0.1) and transparency (alpha=0.5) for better visibility
            
            4. Bounding Box Overlay:
               - Retrieves 3D boxes transformed to sensor coordinate frame
               - BoxVisibility.ANY includes all boxes regardless of visibility status
               - Each box rendered as wireframe by connecting bottom face corners
               - Corners format: 3x8 array (xyz coordinates of 8 box vertices)
               - Draws 4 edges connecting bottom face corners (indices 0,1,2,3)
            
            5. Coordinate System:
               - X-axis: Forward direction (vehicle's driving direction)
               - Y-axis: Left direction (perpendicular to driving direction)
               - Z-axis: Up direction (vertical, perpendicular to ground)
            
            6. Output:
               - Saves high-resolution PNG (150 DPI) with tight bounding box
               - Closes figure to free memory (important in batch processing)
        """
        # Get the sample record and extract LIDAR_TOP sensor token
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']  # Top-mounted LiDAR is primary 3D sensor

        # Load point cloud file from disk
        sample_data = self.nusc.get('sample_data', lidar_token)
        pcl_path = os.path.join(self.nusc.dataroot, sample_data['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        points = pc.points[:3, :]  # Extract only xyz coordinates (discard intensity channel)

        # Subsample points for efficient rendering (typical cloud has ~30K-40K points)
        indices = np.random.choice(points.shape[1],  # Total available points
                                   size=min(10000, points.shape[1]),  # Sample up to 10K
                                   replace=False)  # No duplicates
        points_sampled = points[:, indices]

        # Create 3D matplotlib figure and axis
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Render point cloud as 3D scatter plot
        # Color points by height (z-coordinate) for depth perception
        ax.scatter(points_sampled[0],  # X coordinates
                   points_sampled[1],  # Y coordinates
                   points_sampled[2],  # Z coordinates
                   c=points_sampled[2],  # Color by height
                   cmap='viridis',  # Yellow (high) to purple (low) colormap
                   s=0.1,  # Small point size
                   alpha=0.5)  # Semi-transparent for better overlap visibility

        # Retrieve 3D bounding boxes transformed to LiDAR sensor frame
        # Returns: pointcloud, boxes (in sensor frame), camera_intrinsic
        _, boxes, _ = self.nusc.get_sample_data(lidar_token, 
                                                box_vis_level=BoxVisibility.ANY)

        # Draw wireframe bounding boxes in red
        for box in boxes:
            corners = box.corners()  # Get 3x8 array of box corner coordinates
            
            # Draw bottom face of bounding box (4 edges connecting corners 0,1,2,3)
            for i in [0, 1, 2, 3]:
                j = (i + 1) % 4  # Next corner (wraps 3->0)
                ax.plot([corners[0, i], corners[0, j]],  # X coordinates of edge
                        [corners[1, i], corners[1, j]],  # Y coordinates of edge
                        [corners[2, i], corners[2, j]],  # Z coordinates of edge
                        'r-', linewidth=2)  # Red solid line

        # Set axis labels with units
        ax.set_xlabel('X (m)')  # Forward
        ax.set_ylabel('Y (m)')  # Left
        ax.set_zlabel('Z (m)')  # Up
        ax.set_title('3D LiDAR Scene with Annotations')
        
        # Save figure to disk and clean up
        output_path = self.output_dir / f'3d_scene_{sample_token}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Free memory
        print(f"Saved 3D scene visualization to: {output_path}")

    def list_scenes(self):
        """
        Print a formatted list of all scenes in the dataset to console.
        
        A 'scene' in nuScenes represents a continuous 20-second driving segment.
        This method delegates to nuScenes' built-in list_scenes() which prints:
        - Scene token (unique identifier)
        - Scene name/description
        - Timestamp when recorded
        - Duration in seconds
        - Location (e.g., boston-seaport, singapore-onenorth)
        - Number of annotations in the scene
        
        Technical Details:
            - Uses nuScenes API's formatted output
            - Useful for browsing dataset contents and selecting scenes for analysis
            - v1.0-mini contains 10 scenes (subset of full dataset's 1000 scenes)
        """
        self.nusc.list_scenes()

    def visualize_sample(self):
        """
        Render a multi-sensor visualization of a complete sample (all 6 cameras).
        
        Creates a 2x3 grid showing synchronized images from all 6 cameras mounted on the vehicle
        at a single timestamp. This provides a 360-degree view around the autonomous vehicle.
        
        Technical Details:
            1. Hardcoded to visualize scene index 1 (second scene in dataset)
            2. Uses first sample (timestamp) in that scene
            3. Calls nuScenes' render_sample() which:
               - Loads images from all 6 cameras: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
                 CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
               - Overlays 2D projections of 3D bounding boxes on each camera view
               - Arranges in 2x3 grid (front cameras top row, back cameras bottom row)
            4. Camera coverage:
               - Each camera: ~70° horizontal field of view
               - Combined: Full 360° coverage around vehicle
            5. Saves composite image showing complete sensor suite view
        """
        # Select scene index 1 (hardcoded for demo purposes)
        my_scene = self.nusc.scene[1]
        first_sample_token = my_scene["first_sample_token"]  # Get first timestamp in scene

        # Render all 6 camera views with 2D projected annotations
        output_path = self.output_dir / f'sample_{first_sample_token}.png'
        self.nusc.render_sample(first_sample_token)  # Creates matplotlib figure internally
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Free memory
        print(f"Saved sample visualization to: {output_path}")

    def visualize_sample_data(self):
        """
        Render a single camera image with overlaid 2D bounding box annotations.
        
        Visualizes one specific sensor (front camera) at a specific timestamp, showing
        the raw camera image with 2D projections of 3D object annotations overlaid.
        
        Technical Details:
            1. Sensor Selection:
               - Hardcoded to CAM_FRONT (forward-facing camera)
               - Could be any of the 6 cameras or LIDAR_TOP sensor
            
            2. Data Flow:
               - sample['data'] dict maps sensor names to sample_data tokens
               - sample_data record contains: filename, timestamp, calibration reference
               - Retrieves CAM_FRONT's sample_data token for this timestamp
            
            3. Rendering Process (via nuScenes API):
               - Loads camera image from JPEG file
               - Retrieves all 3D annotations for this sample
               - Projects 3D boxes to 2D image plane using camera intrinsics and extrinsics
               - Draws 2D bounding boxes on image
               - Filters boxes by visibility and frustum culling
            
            4. Projection Math:
               - 3D world coords → ego vehicle frame → sensor frame → image plane
               - Uses camera calibration matrix (intrinsics) and pose (extrinsics)
               - Only renders objects visible in camera's field of view
            
            5. Output: Single annotated camera image saved as PNG
        """
        # Navigate to scene 1, first timestamp
        my_scene = self.nusc.scene[1]
        first_sample_token = my_scene["first_sample_token"]
        my_sample = self.nusc.get("sample", first_sample_token)
        
        # Select front camera sensor
        sensor = 'CAM_FRONT'  # Could be any sensor: CAM_BACK, CAM_FRONT_LEFT, etc.
        cam_front_data = self.nusc.get('sample_data', my_sample['data'][sensor])

        # Render camera image with 2D projected annotations
        output_path = self.output_dir / f'sample_data_{cam_front_data["token"]}.png'
        self.nusc.render_sample_data(cam_front_data['token'])  # Creates matplotlib figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Free memory
        print(f"Saved sample data visualization to: {output_path}")

    def visualize_annotation(self):
        """
        Render individual object annotations across all camera views where visible.
        
        For each annotated object, creates a visualization showing the object's 2D bounding box
        projected onto all camera images where it appears. Demonstrates how a single 3D object
        annotation maps to multiple 2D views.
        
        Technical Details:
            1. Annotation Selection:
               - Processes first 3 annotations from scene 1's first sample
               - Limits to 3 to avoid excessive output (samples can have 50+ annotations)
            
            2. For Each Annotation:
               a) Prints full annotation metadata to console:
                  - token: Unique annotation identifier
                  - sample_token: Parent sample (timestamp) reference
                  - instance_token: Object instance ID (tracks same object across frames)
                  - visibility_token: How well object is visible (1-4 scale)
                  - category_name: Object class (e.g., 'vehicle.car', 'human.pedestrian.adult')
                  - translation: 3D center position [x, y, z] in global coordinates (meters)
                  - size: [width, length, height] of 3D bounding box (meters)
                  - rotation: Quaternion [w, x, y, z] for 3D orientation
                  - num_lidar_pts: Point cloud points inside this box
                  - num_radar_pts: Radar detections for this object
               
               b) Calls nuScenes' render_annotation() which:
                  - Loads images from all 6 cameras
                  - Projects 3D box to 2D in each camera view
                  - Only shows cameras where object is visible (in field of view)
                  - Creates multi-panel figure with relevant camera views
            
            3. Use Cases:
               - Understanding annotation structure and metadata
               - Verifying annotation quality across multiple views
               - Debugging object tracking and visibility
            
            4. Output: One PNG per annotation showing all relevant camera views
        """
        # Navigate to scene 1, first timestamp
        my_scene = self.nusc.scene[1]
        first_sample_token = my_scene["first_sample_token"]
        my_sample = self.nusc.get("sample", first_sample_token)
        annotation_tokens = my_sample['anns']  # List of all annotation tokens for this sample

        # Process first 3 annotations only (for demonstration)
        for idx, annotation_token in enumerate(annotation_tokens[:3]):
            # Print complete annotation metadata to console for inspection
            my_annotation_metadata = self.nusc.get('sample_annotation', annotation_token)
            print(my_annotation_metadata)
            
            # Render this annotation across all cameras where visible
            self.nusc.render_annotation(annotation_token)  # Creates matplotlib figure
            output_path = self.output_dir / f'annotation_{idx}_{annotation_token}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()  # Free memory
            print(f"Saved annotation visualization to: {output_path}")

    def inspect(self, sample):

        self.list_scenes()
        self.visualize_sample()
        self.visualize_sample_data()
        self.visualize_annotation()

        lidar_token = sample['data']['LIDAR_TOP']

        pc_info = self.inspect_point_cloud(lidar_token)
        annotations = self.inspect_annotations(sample['token'])
        self.visualize_3d_scene(sample['token'])

        print("\n=== Inspection Point 1: Raw Data ===")
        print(f"\nPoint Cloud: {pc_info['num_points']} points")
        print(f"X: [{pc_info['x_range'][0]:.2f}, {pc_info['x_range'][1]:.2f}] m")
        print(f"Y: [{pc_info['y_range'][0]:.2f}, {pc_info['y_range'][1]:.2f}] m")
        print(f"Z: [{pc_info['z_range'][0]:.2f}, {pc_info['z_range'][1]:.2f}] m")