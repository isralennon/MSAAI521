import numpy as np
import os
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class RawDataInspector:
    def __init__(self, nusc):
        self.nusc = nusc
        

    def inspect_point_cloud(self, sample_data_token):
        sample_data = self.nusc.get('sample_data', sample_data_token)
        pcl_path = os.path.join(self.nusc.dataroot, sample_data['filename'])

        pc = LidarPointCloud.from_file(pcl_path)
        points = pc.points

        return {
            'shape': points.shape,
            'num_points': points.shape[1],
            'x_range': (points[0].min(), points[0].max()),
            'y_range': (points[1].min(), points[1].max()),
            'z_range': (points[2].min(), points[2].max()),
            'intensity_range': (points[3].min(), points[3].max())
        }

    def inspect_annotations(self, sample_token):
        sample = self.nusc.get('sample', sample_token)

        annotations = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            annotations.append({
                'category': ann['category_name'],
                'translation': ann['translation'],
                'size': ann['size'],
                'rotation': ann['rotation']
            })

        return annotations

    def visualize_3d_scene(self, sample_token):
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']

        sample_data = self.nusc.get('sample_data', lidar_token)
        pcl_path = os.path.join(self.nusc.dataroot, sample_data['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        points = pc.points[:3, :]

        indices = np.random.choice(points.shape[1], 
                                   size=min(10000, points.shape[1]), 
                                   replace=False)
        points_sampled = points[:, indices]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points_sampled[0], 
                   points_sampled[1], 
                   points_sampled[2], 
                   c=points_sampled[2], 
                   cmap='viridis', 
                   s=0.1, 
                   alpha=0.5)

        _, boxes, _ = self.nusc.get_sample_data(lidar_token, 
                                                box_vis_level=BoxVisibility.ANY)

        for box in boxes:
            corners = box.corners()
            for i in [0, 1, 2, 3]:
                j = (i + 1) % 4
                ax.plot([corners[0, i], corners[0, j]],
                        [corners[1, i], corners[1, j]],
                        [corners[2, i], corners[2, j]], 'r-', linewidth=2)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D LiDAR Scene with Annotations')
        plt.show()

    def list_scenes(self):
        self.nusc.list_scenes()

    def visualize_sample(self):
        my_scene = self.nusc.scene[1]
        first_sample_token = my_scene["first_sample_token"]

        self.nusc.render_sample(first_sample_token)

    def visualize_sample_data(self):
        my_scene = self.nusc.scene[1]
        first_sample_token = my_scene["first_sample_token"]
        my_sample = self.nusc.get("sample", first_sample_token)
        sensor = 'CAM_FRONT'
        cam_front_data = self.nusc.get('sample_data', my_sample['data'][sensor])

        self.nusc.render_sample_data(cam_front_data['token'])

    def visualize_annotation(self):
        my_scene = self.nusc.scene[1]
        first_sample_token = my_scene["first_sample_token"]
        my_sample = self.nusc.get("sample", first_sample_token)
        annotation_tokens = my_sample['anns']

        for annotation_token in annotation_tokens[:3]:
            my_annotation_metadata = self.nusc.get('sample_annotation', annotation_token)
            print(my_annotation_metadata)
            self.nusc.render_annotation(annotation_token)
            plt.show()
