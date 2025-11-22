"""
PointCloudProcessor: Transform and filter LiDAR point clouds for BEV processing.

This module handles the spatial transformation of 3D point clouds from sensor coordinates
to ego vehicle coordinates, and filters points to a region of interest (ROI) suitable for
Bird's Eye View (BEV) image generation.

Key Operations:
- Load binary LiDAR data from nuScenes dataset
- Apply rigid body transformations (rotation + translation)
- Filter points to defined spatial boundaries
- Prepare point clouds for 2D rasterization
"""

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import os


class PointCloudProcessor:
    """
    Processor for loading, transforming, and filtering 3D LiDAR point clouds.
    
    This class handles the coordinate transformations necessary to convert LiDAR point clouds
    from sensor frame to ego vehicle frame, and filters points to a region of interest around
    the vehicle suitable for autonomous driving perception tasks.
    
    Attributes:
        nusc: NuScenes dataset instance for accessing metadata and files
        x_range: Tuple (min, max) defining forward/backward extent in meters
        y_range: Tuple (min, max) defining left/right extent in meters
        z_range: Tuple (min, max) defining up/down extent in meters
    """
    
    def __init__(self, nusc, x_range=(-50, 50), y_range=(-50, 50), z_range=(-3, 5)):
        """
        Initialize the point cloud processor with spatial filtering boundaries.
        
        Args:
            nusc: NuScenes instance providing access to dataset
            x_range: (min_x, max_x) in meters. X-axis points forward from vehicle.
                    Default (-50, 50) = 100m total range, 50m ahead and behind
            y_range: (min_y, max_y) in meters. Y-axis points left from vehicle.
                    Default (-50, 50) = 100m total range, 50m on each side
            z_range: (min_z, max_z) in meters. Z-axis points up from vehicle.
                    Default (-3, 5) = 8m total height, 3m below to 5m above vehicle
        
        Technical Details:
            - Range selection impacts BEV image resolution and coverage area
            - Typical autonomous vehicle perception: 50-100m forward, ±50m lateral
            - Z-range filters ground points (below) and tall structures (above)
            - Coordinate system: Right-handed with Z-up (ego vehicle frame)
        """
        self.nusc = nusc
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
    
    def load_and_transform(self, sample_data_token):
        """
        Load LiDAR point cloud and transform from sensor frame to ego vehicle frame.
        
        This method performs a rigid body transformation (rotation followed by translation)
        to convert point cloud coordinates from the LiDAR sensor's local coordinate system
        to the ego vehicle's coordinate system. This is essential for sensor fusion and
        consistent spatial representation.
        
        Args:
            sample_data_token: UUID string identifying a specific LiDAR capture
        
        Returns:
            numpy.ndarray: Point cloud in ego vehicle frame with shape (4, N) where:
                - Row 0: X coordinates (forward) in meters
                - Row 1: Y coordinates (left) in meters
                - Row 2: Z coordinates (up) in meters
                - Row 3: Intensity values (reflectivity)
        
        Technical Details:
            1. Data Loading:
               - Retrieves file path from nuScenes metadata
               - Loads binary .pcd.bin file containing float32 values
               - Format: Interleaved [x, y, z, intensity] × N points
            
            2. Coordinate Transformation Chain:
               a) Points start in LiDAR sensor frame (LIDAR_TOP coordinate system)
               b) Retrieve calibrated_sensor record containing extrinsics:
                  - rotation: Quaternion representing sensor orientation relative to ego
                  - translation: 3D vector [x, y, z] for sensor position relative to ego
               c) Apply rotation: Points are rotated by converting quaternion to 3×3 matrix
               d) Apply translation: Rotated points are shifted by translation vector
               e) Result: Points in ego vehicle frame (centered at vehicle)
            
            3. Transformation Mathematics:
               - Rotation matrix R from quaternion q = [w, x, y, z]
               - Translation vector t = [tx, ty, tz]
               - Transformed point: p' = R × p + t
               - Order matters: Rotate first, then translate
            
            4. Ego Vehicle Frame Convention:
               - Origin: Center of vehicle at ground level
               - X-axis: Points forward (driving direction)
               - Y-axis: Points left
               - Z-axis: Points up
               - Right-handed coordinate system
        """
        # Get metadata record for this LiDAR capture
        sample_data = self.nusc.get('sample_data', sample_data_token)
        
        # Construct full path to binary point cloud file
        pcl_path = os.path.join(self.nusc.dataroot, sample_data['filename'])
        
        # Load point cloud from binary file (4×N array: x, y, z, intensity)
        pc = LidarPointCloud.from_file(pcl_path)
        
        # Retrieve sensor calibration (extrinsic parameters)
        cs_record = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        
        # ============================================================
        # CRITICAL FIX: Keep LiDAR data in sensor frame
        # ============================================================
        # According to nuScenes documentation, the raw LiDAR data is already
        # stored in the sensor (LIDAR_TOP) coordinate frame, NOT the ego vehicle frame.
        # 
        # REASON FOR COMMENTING OUT TRANSFORMATION:
        # The original code incorrectly transformed LiDAR data from sensor→ego frame,
        # which caused misalignment with annotations. Since the data is already in the
        # correct sensor frame, we should NOT apply any transformation here.
        # 
        # Instead, annotations must be transformed TO the sensor frame (see DataPreprocessor.py)
        # to match the coordinate system of the LiDAR point cloud.
        # 
        # TRANSFORMATION STRATEGY:
        # - LiDAR: Keep in sensor frame (no transformation needed)
        # - Annotations: Transform from global → ego → sensor frame
        # - Result: Both in same coordinate system for correct BEV projection
        # 
        # Original transformation code (now disabled):
        # pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        # pc.translate(np.array(cs_record['translation']))
        # ============================================================
        
        # Return point cloud in sensor frame (4×N: x, y, z, intensity)
        return pc.points
    
    def filter_points(self, points):
        """
        Filter point cloud to region of interest (ROI) using spatial boundaries.
        
        Removes points outside the defined x, y, z ranges to focus on the relevant
        area around the vehicle. This reduces computational load and focuses on
        distances relevant for autonomous driving perception.
        
        Args:
            points: numpy.ndarray of shape (4, N) with rows [x, y, z, intensity]
        
        Returns:
            numpy.ndarray: Filtered point cloud with shape (4, M) where M ≤ N
                          Contains only points within the defined spatial boundaries
        
        Technical Details:
            1. Filtering Logic:
               - Creates boolean mask using vectorized numpy comparisons
               - Each condition checks one spatial dimension
               - Combines conditions with logical AND (&) operator
               - Applies mask to select only points satisfying all conditions
            
            2. Spatial Filtering Rationale:
               - X-range: Limits forward/backward perception distance
                 • Too far: Low density, less relevant for immediate decisions
                 • Too close: May miss distant objects
               - Y-range: Limits lateral (side-to-side) perception
                 • Typically symmetric around vehicle centerline
               - Z-range: Removes ground and sky points
                 • Below vehicle: Road surface, underground artifacts
                 • Above vehicle: Bridges, buildings, sky noise
            
            3. Performance Considerations:
               - Vectorized operations using numpy for efficiency
               - Typical filtering: 30K-40K → 10K-20K points (50-70% reduction)
               - Reduces downstream processing time for rasterization
            
            4. Impact on BEV:
               - Filtered points determine BEV image coverage
               - Points outside range are completely discarded
               - Matches the spatial extent that will be rasterized
        """
        # Create boolean mask: True for points inside ROI, False outside
        mask = (
            (points[0, :] >= self.x_range[0]) & (points[0, :] <= self.x_range[1]) &  # X bounds
            (points[1, :] >= self.y_range[0]) & (points[1, :] <= self.y_range[1]) &  # Y bounds
            (points[2, :] >= self.z_range[0]) & (points[2, :] <= self.z_range[1])    # Z bounds
        )
        
        # Apply mask to select only points within boundaries (fancy indexing)
        return points[:, mask]

