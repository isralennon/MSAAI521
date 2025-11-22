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

        # Color mapping to match 2D BEV visualization
        color_map = {
            'vehicle.car': '#00FF00',           # Green
            'vehicle.taxi': '#00FF00',
            'vehicle.truck': '#00FF00',         # Green (Truck/Bus category)
            'vehicle.bus.bendy': '#00FF00',
            'vehicle.bus.rigid': '#00FF00',
            'vehicle.construction': '#00FF00',
            'human.pedestrian.adult': '#FF0000',                # Red
            'human.pedestrian.child': '#FF0000',
            'human.pedestrian.construction_worker': '#FF0000',
            'human.pedestrian.police_officer': '#FF0000',
            'vehicle.bicycle': '#0000FF',       # Blue (Cyclist)
            'vehicle.motorcycle': '#0000FF',
            'movable_object.barrier': '#FFFF00',            # Yellow
            'movable_object.trafficcone': '#FFA500',        # Orange
            'movable_object.pushable_pullable': '#FF00FF',  # Magenta
        }
        
        def get_color(box_name):
            return color_map.get(box_name, '#FFFFFF')  # White for unmapped categories

        print(f"\n3D Scene - Found {len(boxes)} annotations:")
        for idx, box in enumerate(boxes):
            corners = box.corners()
            color = get_color(box.name)
            
            # Draw bottom face
            for i in [0, 1, 2, 3]:
                j = (i + 1) % 4
                ax.plot([corners[0, i], corners[0, j]],
                        [corners[1, i], corners[1, j]],
                        [corners[2, i], corners[2, j]], color=color, linewidth=2)
            
            # Draw vertical edges
            for i in range(4):
                ax.plot([corners[0, i], corners[0, i+4]],
                        [corners[1, i], corners[1, i+4]],
                        [corners[2, i], corners[2, i+4]], color=color, linewidth=2)
            
            # Draw top face
            for i in [4, 5, 6, 7]:
                j = 4 + ((i + 1) % 4)
                ax.plot([corners[0, i], corners[0, j]],
                        [corners[1, i], corners[1, j]],
                        [corners[2, i], corners[2, j]], color=color, linewidth=2)
            
            center = box.center
            print(f"  [{idx}] {box.name}: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), size={box.wlh}")

        # Create legend with unique categories and their colors
        unique_names = sorted(list(set([box.name for box in boxes])))
        legend_handles = [plt.Line2D([0], [0], color=get_color(name), linewidth=2, label=name) 
                         for name in unique_names]
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D LiDAR Scene with {len(boxes)} Annotations')
        ax.legend(handles=legend_handles, loc='upper left', fontsize=8, 
                 bbox_to_anchor=(1.05, 1), ncol=1)
        plt.tight_layout()
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
