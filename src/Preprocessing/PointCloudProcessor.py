import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import os


class PointCloudProcessor:
    def __init__(self, nusc, x_range=(-50, 50), y_range=(-50, 50), z_range=(-3, 5)):
        self.nusc = nusc
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
    
    def load_and_transform(self, sample_data_token):
        sample_data = self.nusc.get('sample_data', sample_data_token)
        pcl_path = os.path.join(self.nusc.dataroot, sample_data['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        
        cs_record = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))
        
        return pc.points
    
    def filter_points(self, points):
        mask = (
            (points[0, :] >= self.x_range[0]) & (points[0, :] <= self.x_range[1]) &
            (points[1, :] >= self.y_range[0]) & (points[1, :] <= self.y_range[1]) &
            (points[2, :] >= self.z_range[0]) & (points[2, :] <= self.z_range[1])
        )
        return points[:, mask]

