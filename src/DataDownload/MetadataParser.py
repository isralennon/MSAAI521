from nuscenes.nuscenes import NuScenes
from Globals import NUSCENES_ROOT, NUSCENES_VERSION


class MetadataParser:
    def __init__(self, root_path=NUSCENES_ROOT, version=NUSCENES_VERSION):
        self.root_path = root_path
        self.version = version
        self.nusc = None
    
    def parse(self):
        self.nusc = NuScenes(version=self.version, dataroot=self.root_path, verbose=False)
        return self.nusc

