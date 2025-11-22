from Preprocessing.PointCloudProcessor import PointCloudProcessor
from Preprocessing.BEVRasterizer import BEVRasterizer
from Preprocessing.YOLOAnnotationConverter import YOLOAnnotationConverter
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import numpy as np
import cv2
import os
from pathlib import Path
from Globals import PREPROCESSED_ROOT


class DataPreprocessor:
    def __init__(self, nusc):
        self.nusc = nusc
        self.pc_processor = PointCloudProcessor(nusc)
        self.rasterizer = BEVRasterizer()
        self.converter = YOLOAnnotationConverter(self.rasterizer.width, self.rasterizer.height)
        
        self.output_root = Path(PREPROCESSED_ROOT)
        self.images_dir = self.output_root / 'images'
        self.labels_dir = self.output_root / 'labels'
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
    
    def process_all_samples(self, debug_first=False):
        total_samples = len(self.nusc.sample)
        
        for sample_idx in range(total_samples):
            sample = self.nusc.sample[sample_idx]
            lidar_token = sample['data']['LIDAR_TOP']
            
            points = self.pc_processor.load_and_transform(lidar_token)
            filtered_points = self.pc_processor.filter_points(points)
            bev_image = self.rasterizer.rasterize(filtered_points)
            
            sample_data = self.nusc.get('sample_data', lidar_token)
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            
            yolo_labels = []
            calibrated_sensor = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            
            if debug_first and sample_idx == 0:
                print(f"\n{'='*60}")
                print(f"DEBUG: Processing sample {sample_idx}")
                print(f"{'='*60}")
                print(f"Filtered points shape: {filtered_points.shape}")
                print(f"Point cloud X range: [{filtered_points[0].min():.2f}, {filtered_points[0].max():.2f}]")
                print(f"Point cloud Y range: [{filtered_points[1].min():.2f}, {filtered_points[1].max():.2f}]")
                print(f"Point cloud Z range: [{filtered_points[2].min():.2f}, {filtered_points[2].max():.2f}]")
                print(f"\nProcessing {len(sample['anns'])} annotations:")
            
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                
                # Create box in global coordinates
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                
                # Transform from global to ego vehicle coordinates
                box.translate(-np.array(ego_pose['translation']))
                box.rotate(Quaternion(ego_pose['rotation']).inverse)
                
                # Transform from ego vehicle to sensor coordinates
                box.translate(-np.array(calibrated_sensor['translation']))
                box.rotate(Quaternion(calibrated_sensor['rotation']).inverse)
                
                # Get box corners to compute axis-aligned bounding box in BEV
                corners = box.corners()  # 3x8 array of corner coordinates
                
                # For BEV, we only care about X and Y (top-down view)
                x_corners = corners[0, :]  # X coordinates of all 8 corners
                y_corners = corners[1, :]  # Y coordinates of all 8 corners
                
                # Compute axis-aligned bounding box
                x_min, x_max = x_corners.min(), x_corners.max()
                y_min, y_max = y_corners.min(), y_corners.max()
                
                # Center and size of the axis-aligned box
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                length = y_max - y_min
                
                yolo_label = self.converter.convert_annotation(
                    np.array([x_center, y_center, box.center[2]]),
                    np.array([width, length, box.wlh[2]]),
                    ann['category_name'],
                    debug=debug_first and sample_idx == 0
                )
                if yolo_label:
                    yolo_labels.append(yolo_label)
            
            scene_name = self.nusc.get('scene', sample['scene_token'])['name']
            filename = f"{scene_name}_{sample['token']}"
            
            image_path = self.images_dir / f"{filename}.png"
            cv2.imwrite(str(image_path), bev_image)
            
            label_path = self.labels_dir / f"{filename}.txt"
            with open(label_path, 'w') as f:
                for label in yolo_labels:
                    f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")
        
        return total_samples

