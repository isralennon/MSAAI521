from pathlib import Path
from Globals import NUSCENES_ROOT


class DataDownloader:
    def __init__(self, root_path=NUSCENES_ROOT):
        self.root = Path(root_path)
    
    def check_and_prompt(self):
        if self.root.exists():
            return True
        
        print(f"nuScenes dataset not found at: {self.root}")
        print()
        print("Download instructions:")
        print("1. Visit: https://www.nuscenes.org/nuscenes#download")
        print("2. Download v1.0-mini (4 GB)")
        print(f"3. Extract to: {self.root}")
        print()
        
        return False