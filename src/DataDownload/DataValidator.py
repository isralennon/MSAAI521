from pathlib import Path
from Globals import NUSCENES_ROOT, NUSCENES_VERSION


class DataValidator:
    def __init__(self, root_path=NUSCENES_ROOT, version=NUSCENES_VERSION):
        self.root = Path(root_path)
        self.version = version
        
        self.required_dirs = [
            f'samples/LIDAR_TOP',
            f'sweeps/LIDAR_TOP',
            self.version
        ]
        
        self.required_files = [
            f'{self.version}/sample.json',
            f'{self.version}/sample_data.json',
            f'{self.version}/sample_annotation.json',
            f'{self.version}/ego_pose.json',
            f'{self.version}/calibrated_sensor.json',
            f'{self.version}/scene.json',
            f'{self.version}/instance.json',
            f'{self.version}/category.json'
        ]
    
    def validate(self):
        if not self.root.exists():
            return False
        
        for dir_path in self.required_dirs:
            if not (self.root / dir_path).exists():
                return False
        
        for file_path in self.required_files:
            if not (self.root / file_path).exists():
                return False
        
        return True

