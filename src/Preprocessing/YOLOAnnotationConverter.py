import numpy as np


class YOLOAnnotationConverter:
    def __init__(self, image_width, image_height, x_range=(-75, 75), y_range=(-75, 75), resolution=0.1):
        self.image_width = image_width
        self.image_height = image_height
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        self.class_mapping = {
            'vehicle.car': 0,
            'vehicle.taxi': 0,
            'vehicle.truck': 1,
            'vehicle.bus.bendy': 1,
            'vehicle.bus.rigid': 1,
            'vehicle.construction': 1,
            'human.pedestrian.adult': 2,
            'human.pedestrian.child': 2,
            'human.pedestrian.construction_worker': 2,
            'human.pedestrian.police_officer': 2,
            'vehicle.bicycle': 3,
            'vehicle.motorcycle': 3
        }
    
    def convert_annotation(self, box_translation, box_size, category_name, debug=False):
        if category_name not in self.class_mapping:
            return None
        
        class_id = self.class_mapping[category_name]
        
        # Box size in nuScenes: [width (x), length (y), height (z)]
        # For BEV top-down view: use width and length
        x_center = (box_translation[0] - self.x_range[0]) / self.resolution
        y_center = (box_translation[1] - self.y_range[0]) / self.resolution
        y_center = self.image_height - 1 - y_center
        
        # box_size[0] = width (dimension along x-axis)
        # box_size[1] = length (dimension along y-axis)
        width = box_size[0] / self.resolution
        height = box_size[1] / self.resolution
        
        x_norm = x_center / self.image_width
        y_norm = y_center / self.image_height
        w_norm = width / self.image_width
        h_norm = height / self.image_height
        
        if debug:
            print(f"  Category: {category_name}, Box translation: {box_translation}, Box size: {box_size}")
            print(f"  Pixel pos: ({x_center:.1f}, {y_center:.1f}), Pixel size: ({width:.1f}, {height:.1f})")
            print(f"  Normalized: center=({x_norm:.3f}, {y_norm:.3f}), size=({w_norm:.3f}, {h_norm:.3f})")
        
        if w_norm <= 0 or h_norm <= 0 or w_norm > 1 or h_norm > 1:
            if debug:
                print(f"  ❌ REJECTED: Invalid box size")
            return None
        
        if x_norm < 0 or x_norm > 1 or y_norm < 0 or y_norm > 1:
            if debug:
                print(f"  ❌ REJECTED: Box center out of bounds")
            return None
        
        if debug:
            print(f"  ✅ ACCEPTED")
        
        return [class_id, 
                np.clip(x_norm, 0, 1), 
                np.clip(y_norm, 0, 1), 
                np.clip(w_norm, 0, 1), 
                np.clip(h_norm, 0, 1)]

