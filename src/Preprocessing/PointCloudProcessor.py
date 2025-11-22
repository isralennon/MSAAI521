import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import os


class PointCloudProcessor:
    def __init__(self, nusc, x_range=(-75, 75), y_range=(-75, 75), z_range=(-3, 5)):
        self.nusc = nusc
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
    
    def load_and_transform(self, sample_data_token):
        sample_data = self.nusc.get('sample_data', sample_data_token)
        pcl_path = os.path.join(self.nusc.dataroot, sample_data['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        
        # LiDAR data is already in sensor frame
        # We keep it in sensor frame for BEV (no transformation needed)
        # Annotations will be transformed TO sensor frame to match
        
        return pc.points
    
    def filter_points(self, points):
        mask = (
            (points[0, :] >= self.x_range[0]) & (points[0, :] <= self.x_range[1]) &
            (points[1, :] >= self.y_range[0]) & (points[1, :] <= self.y_range[1]) &
            (points[2, :] >= self.z_range[0]) & (points[2, :] <= self.z_range[1])
        )
        return points[:, mask]

