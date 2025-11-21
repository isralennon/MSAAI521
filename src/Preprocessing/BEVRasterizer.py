import numpy as np


class BEVRasterizer:
    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), z_range=(-3, 5), resolution=0.1):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution
        self.width = int((x_range[1] - x_range[0]) / resolution)
        self.height = int((y_range[1] - y_range[0]) / resolution)
    
    def rasterize(self, points):
        height_map = np.zeros((self.height, self.width), dtype=np.float32)
        intensity_map = np.zeros((self.height, self.width), dtype=np.float32)
        density_map = np.zeros((self.height, self.width), dtype=np.int32)
        
        x_img = np.int32((points[0, :] - self.x_range[0]) / self.resolution)
        y_img = np.int32((points[1, :] - self.y_range[0]) / self.resolution)
        x_img = np.clip(x_img, 0, self.width - 1)
        y_img = np.clip(y_img, 0, self.height - 1)
        y_img = self.height - 1 - y_img
        
        for i in range(points.shape[1]):
            x, y = x_img[i], y_img[i]
            height_map[y, x] = max(height_map[y, x], points[2, i])
            intensity_map[y, x] += points[3, i]
            density_map[y, x] += 1
        
        mask = density_map > 0
        intensity_map[mask] = intensity_map[mask] / density_map[mask]
        
        height_map = np.clip((height_map - self.z_range[0]) / (self.z_range[1] - self.z_range[0]), 0, 1)
        height_map = np.power(height_map, 0.5)
        
        intensity_map = intensity_map / max(intensity_map.max(), 1e-6)
        intensity_map = np.power(intensity_map, 0.5)
        
        density_norm = np.log1p(density_map.astype(np.float32))
        density_norm = density_norm / max(density_norm.max(), 1e-6)
        density_norm = np.power(density_norm, 0.3)
        
        bev_image = np.stack([height_map, intensity_map, density_norm], axis=-1)
        return (bev_image * 255).astype(np.uint8)

