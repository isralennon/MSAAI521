import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import os
from Globals import PREPROCESSED_ROOT


class BEVInspector:
    def __init__(self):
        self.class_names = ['Car', 'Truck/Bus', 'Pedestrian', 'Cyclist']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        self.images_dir = Path(PREPROCESSED_ROOT) / 'images'
        self.labels_dir = Path(PREPROCESSED_ROOT) / 'labels'
    
    def load_samples(self, num_samples):
        image_files = sorted(list(self.images_dir.glob('*.png')))
        
        step = max(1, len(image_files) // num_samples)
        selected_files = [image_files[i * step] for i in range(num_samples) if i * step < len(image_files)]
        
        bev_images = []
        yolo_labels_list = []
        
        for image_path in selected_files:
            bev_image = cv2.imread(str(image_path))
            
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            yolo_labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            yolo_labels.append([int(parts[0])] + [float(x) for x in parts[1:]])
            
            bev_images.append(bev_image)
            yolo_labels_list.append(yolo_labels)
        
        return bev_images, yolo_labels_list
    
    def visualize(self, bev_image, yolo_labels):
        img_rgb = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        for label in yolo_labels:
            class_id = int(label[0])
            x_center, y_center = label[1] * w, label[2] * h
            box_w, box_h = label[3] * w, label[4] * h
            
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)
            
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), self.colors[class_id], 2)
            cv2.putText(img_rgb, self.class_names[class_id], (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[class_id], 2)
        
        plt.figure(figsize=(12, 12))
        plt.imshow(img_rgb)
        plt.title('BEV Image with YOLO Annotations')
        plt.axis('off')
        plt.show()
    
    def visualize_grid(self, bev_images, yolo_labels_list, num_cols=2):
        return self._draw_grid(bev_images, yolo_labels_list, num_cols)
    
    def _draw_grid(self, bev_images, yolo_labels_list, num_cols=2):
        num_samples = len(bev_images)
        num_rows = (num_samples + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 10 * num_rows))
        if num_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(num_samples):
            img = bev_images[i].copy()
            h, w = img.shape[:2]
            
            for label in yolo_labels_list[i]:
                class_id = int(label[0])
                x_center, y_center = label[1] * w, label[2] * h
                box_w, box_h = label[3] * w, label[4] * h
                
                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), self.colors[class_id], 2)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(f'Sample {i} ({len(yolo_labels_list[i])} objects)')
            axes[i].axis('off')
        
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

