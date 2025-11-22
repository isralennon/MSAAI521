"""
BEVRasterizer: Convert 3D LiDAR point clouds to 2D Bird's Eye View (BEV) images.

This module implements the core rasterization algorithm that projects 3D point clouds
onto a 2D grid viewed from above (bird's eye perspective). The resulting BEV images
encode height, intensity, and density information in three channels, making them
suitable for 2D object detection models.

Key Concepts:
- Orthographic projection from 3D to 2D (top-down view)
- Multi-channel encoding: height, intensity, density
- Pixel-space quantization and accumulation
- Normalization and perceptual enhancement
"""

import numpy as np


class BEVRasterizer:
    """
    Rasterizer that converts 3D point clouds into 2D Bird's Eye View (BEV) images.
    
    This class implements a projection algorithm that creates a top-down view of the
    environment around the vehicle, encoding 3D information into a 2D image format
    compatible with standard 2D object detection architectures like YOLO.
    
    The BEV representation has several advantages:
    - Preserves spatial relationships and distances (unlike perspective images)
    - Eliminates scale variation with distance
    - Provides consistent object sizes regardless of distance
    - Suitable for accurate localization and planning
    
    Attributes:
        x_range: Tuple (min, max) for forward/backward extent (meters)
        y_range: Tuple (min, max) for left/right extent (meters)
        z_range: Tuple (min, max) for height extent (meters)
        resolution: Meters per pixel in BEV image
        width: Image width in pixels
        height: Image height in pixels
    """
    
    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), z_range=(-3, 5), resolution=0.1):
        """
        Initialize BEV rasterizer with spatial parameters.
        
        Args:
            x_range: (min, max) coverage in forward direction (meters)
                    Default: (-50, 50) = 100m range
            y_range: (min, max) coverage in lateral direction (meters)
                    Default: (-50, 50) = 100m range
            z_range: (min, max) height range for normalization (meters)
                    Default: (-3, 5) = 8m height range
            resolution: Spatial resolution in meters per pixel
                       Default: 0.1m = 10cm per pixel
        
        Technical Details:
            1. Resolution Trade-offs:
               - Smaller (e.g., 0.05m): Higher detail, larger images, more memory
               - Larger (e.g., 0.2m): Lower detail, smaller images, faster processing
               - 0.1m (10cm) is common balance for autonomous driving
            
            2. Image Dimensions:
               - Width = (x_range[1] - x_range[0]) / resolution
               - Height = (y_range[1] - y_range[0]) / resolution
               - Example: 100m range / 0.1m resolution = 1000 pixels
               - Typical BEV image: 1000×1000 pixels for 100m×100m area
            
            3. Memory Footprint:
               - 1000×1000×3 channels × 1 byte = ~3MB per BEV image
               - Batch processing requires careful memory management
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution
        
        # Calculate image dimensions from spatial extent and resolution
        self.width = int((x_range[1] - x_range[0]) / resolution)   # Pixels in X direction
        self.height = int((y_range[1] - y_range[0]) / resolution)  # Pixels in Y direction
    
    def rasterize(self, points):
        """
        Convert 3D point cloud to 2D Bird's Eye View image with three channels.
        
        This method implements the core rasterization algorithm that projects 3D points
        onto a 2D grid from above, accumulating height, intensity, and density information
        for each pixel. The result is a 3-channel image encoding the 3D scene from above.
        
        Args:
            points: numpy.ndarray of shape (4, N) containing:
                   - Row 0: X coordinates (forward) in meters
                   - Row 1: Y coordinates (left) in meters
                   - Row 2: Z coordinates (height) in meters
                   - Row 3: Intensity values (LiDAR reflectivity)
        
        Returns:
            numpy.ndarray: BEV image of shape (height, width, 3) with dtype uint8
                          - Channel 0 (Red): Height map (maximum Z value per pixel)
                          - Channel 1 (Green): Intensity map (average intensity per pixel)
                          - Channel 2 (Blue): Density map (point count per pixel)
                          All values normalized to range [0, 255]
        
        Technical Details:
            
            **Algorithm Overview:**
            1. Initialize three 2D accumulation maps (height, intensity, density)
            2. Project 3D points to 2D pixel coordinates
            3. Accumulate values for each pixel
            4. Normalize and enhance channels
            5. Stack into 3-channel RGB image
            
            **Channel Semantics:**
            
            • Height Map (Red Channel):
              - Encodes maximum elevation at each (x,y) location
              - Helps distinguish objects from ground plane
              - Used for detecting vertical structures (vehicles, pedestrians)
              - Formula: max(z) for all points projecting to same pixel
            
            • Intensity Map (Green Channel):
              - Encodes average LiDAR reflectivity
              - Different materials have different reflectivities
              - Helps distinguish object types (metal vs. fabric vs. vegetation)
              - Formula: mean(intensity) for points in each pixel
            
            • Density Map (Blue Channel):
              - Encodes number of LiDAR points per pixel
              - Indicates measurement confidence and proximity
              - Higher density = closer objects or better visibility
              - Formula: count(points) per pixel, log-normalized
            
            **Coordinate Transformation:**
            - 3D world coordinates (meters) → 2D pixel coordinates
            - X_pixel = (X_world - X_min) / resolution
            - Y_pixel = (Y_world - Y_min) / resolution
            - Y-axis is flipped (image origin at top-left, world origin at center)
            
            **Normalization Strategy:**
            - Height: Linear normalization to [0,1], then sqrt for perceptual balance
            - Intensity: Linear normalization to [0,1], then sqrt for contrast
            - Density: Log normalization (log1p) for wide range, then power 0.3
            - All channels scaled to [0, 255] for uint8 image format
        """
        # Initialize three 2D accumulation maps for the BEV image
        height_map = np.zeros((self.height, self.width), dtype=np.float32)     # Max height per pixel
        intensity_map = np.zeros((self.height, self.width), dtype=np.float32)  # Sum of intensities
        density_map = np.zeros((self.height, self.width), dtype=np.int32)      # Point count per pixel
        
        # ============================================================
        # STEP 1: Project 3D points to 2D pixel coordinates
        # ============================================================
        
        # Convert world coordinates (meters) to pixel coordinates
        # X direction: forward/backward in world → horizontal in image
        x_img = np.int32((points[0, :] - self.x_range[0]) / self.resolution)
        
        # Y direction: left/right in world → vertical in image
        y_img = np.int32((points[1, :] - self.y_range[0]) / self.resolution)
        
        # Clamp pixel coordinates to image boundaries (handles edge cases)
        x_img = np.clip(x_img, 0, self.width - 1)
        y_img = np.clip(y_img, 0, self.height - 1)
        
        # Flip Y-axis: image origin is top-left, world origin is center-bottom
        # This makes the image appear with vehicle at bottom, forward direction up
        y_img = self.height - 1 - y_img
        
        # ============================================================
        # STEP 2: Accumulate values for each pixel
        # ============================================================
        
        # Iterate through all points and update the three maps
        # Note: Could be optimized with numpy operations, but loop is clear
        for i in range(points.shape[1]):
            x, y = x_img[i], y_img[i]  # Pixel coordinates for this point
            
            # Height map: Keep maximum Z value (tallest point at this location)
            height_map[y, x] = max(height_map[y, x], points[2, i])
            
            # Intensity map: Accumulate intensity (will average later)
            intensity_map[y, x] += points[3, i]
            
            # Density map: Count number of points
            density_map[y, x] += 1
        
        # ============================================================
        # STEP 3: Compute average intensity per pixel
        # ============================================================
        
        # Create mask for pixels with at least one point
        mask = density_map > 0
        
        # Convert accumulated intensity sum to average intensity
        # Only for pixels with points (avoid division by zero)
        intensity_map[mask] = intensity_map[mask] / density_map[mask]
        
        # ============================================================
        # STEP 4: Normalize and enhance each channel
        # ============================================================
        
        # --- Height Map Normalization ---
        # Convert from absolute height (meters) to normalized [0, 1] range
        # Formula: (height - min) / (max - min)
        height_map = np.clip((height_map - self.z_range[0]) / (self.z_range[1] - self.z_range[0]), 0, 1)
        
        # Apply gamma correction (power 0.5 = square root) for perceptual enhancement
        # Brightens darker values, compresses brighter values
        # Helps distinguish low-height objects from ground
        height_map = np.power(height_map, 0.5)
        
        # --- Intensity Map Normalization ---
        # Normalize to [0, 1] range based on maximum intensity in this frame
        # Add small epsilon (1e-6) to avoid division by zero
        intensity_map = intensity_map / max(intensity_map.max(), 1e-6)
        
        # Apply gamma correction for contrast enhancement
        # Makes subtle reflectivity differences more visible
        intensity_map = np.power(intensity_map, 0.5)
        
        # --- Density Map Normalization ---
        # Use logarithmic normalization for wide dynamic range
        # log1p(x) = log(1 + x) handles zero values gracefully
        # Compresses high densities while preserving low density variation
        density_norm = np.log1p(density_map.astype(np.float32))
        
        # Scale to [0, 1] range
        density_norm = density_norm / max(density_norm.max(), 1e-6)
        
        # Apply strong power transformation (0.3) to further compress range
        # Emphasizes presence of points over exact count
        density_norm = np.power(density_norm, 0.3)
        
        # ============================================================
        # STEP 5: Stack channels and convert to uint8 image format
        # ============================================================
        
        # Stack three normalized maps into RGB image (height, width, 3)
        # Channel order: [height, intensity, density] → [R, G, B]
        bev_image = np.stack([height_map, intensity_map, density_norm], axis=-1)
        
        # Scale from [0, 1] float to [0, 255] uint8 for standard image format
        return (bev_image * 255).astype(np.uint8)

