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
    
    def process_all_samples(self):
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
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                box.translate(-np.array(ego_pose['translation']))
                box.rotate(Quaternion(ego_pose['rotation']).inverse)
                
                yolo_label = self.converter.convert_annotation(
                    box.center,
                    box.wlh,
                    ann['category_name']
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

