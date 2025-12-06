"""
GPU-Accelerated BEV Rasterizer using PyTorch

This module provides a drop-in replacement for the CPU-based BEVRasterizer,
using PyTorch CUDA for 10-20x speedup on GPU hardware.

Advantages over CuPy version:
- Uses PyTorch which is already installed
- No additional dependencies
- Compatible with existing CUDA setup

Performance comparison (1280x1280 image):
- CPU (NumPy): ~800ms per frame
- GPU (PyTorch): ~40-80ms per frame
"""

import numpy as np
import torch


class BEVRasterizerTorch:
    """
    GPU-accelerated Bird's Eye View rasterizer using PyTorch.
    
    This class provides the same API as the original BEVRasterizer but uses
    GPU operations for 10-20x speedup. Automatically falls back to CPU if
    GPU is not available.
    
    Attributes:
        x_range: Tuple (min, max) for forward/backward extent (meters)
        y_range: Tuple (min, max) for left/right extent (meters)
        z_range: Tuple (min, max) for height extent (meters)
        resolution: Meters per pixel in BEV image
        width: Image width in pixels
        height: Image height in pixels
        use_gpu: Whether GPU is being used
        device: torch.device ('cuda' or 'cpu')
    """
    
    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), z_range=(-3, 5), resolution=0.1):
        """
        Initialize BEV rasterizer with spatial parameters.
        
        Args:
            x_range: (min, max) coverage in forward direction (meters)
            y_range: (min, max) coverage in lateral direction (meters)
            z_range: (min, max) height range for normalization (meters)
            resolution: Spatial resolution in meters per pixel
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution
        
        # Calculate image dimensions
        self.width = int((x_range[1] - x_range[0]) / resolution)
        self.height = int((y_range[1] - y_range[0]) / resolution)
        
        # Check GPU availability
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        if self.use_gpu:
            # Pre-allocate reusable buffers on GPU for efficiency
            self.height_map = torch.zeros((self.height, self.width), dtype=torch.float32, device=self.device)
            self.intensity_sum = torch.zeros((self.height, self.width), dtype=torch.float32, device=self.device)
            self.density_map = torch.zeros((self.height, self.width), dtype=torch.int32, device=self.device)
    
    def rasterize(self, points):
        """
        Convert 3D point cloud to 2D Bird's Eye View image (GPU-accelerated).
        
        Args:
            points: numpy.ndarray of shape (4, N) containing:
                   - Row 0: X coordinates (forward) in meters
                   - Row 1: Y coordinates (left) in meters
                   - Row 2: Z coordinates (height) in meters
                   - Row 3: Intensity values (LiDAR reflectivity)
        
        Returns:
            numpy.ndarray: BEV image of shape (height, width, 3) with dtype uint8
                          - Channel 0 (Red): Height map
                          - Channel 1 (Green): Intensity map
                          - Channel 2 (Blue): Density map
        """
        if self.use_gpu:
            return self._rasterize_gpu(points)
        else:
            return self._rasterize_cpu(points)
    
    def _rasterize_gpu(self, points):
        """GPU implementation using PyTorch with optimized scatter operations."""
        # Transfer points to GPU
        if isinstance(points, np.ndarray):
            points_gpu = torch.from_numpy(points).to(self.device)
        else:
            points_gpu = points.to(self.device)
        
        # Reset accumulation maps (faster than creating new ones)
        self.height_map.zero_()
        self.intensity_sum.zero_()
        self.density_map.zero_()
        
        # Project 3D points to 2D pixel coordinates
        x_img = ((points_gpu[0, :] - self.x_range[0]) / self.resolution).long()
        y_img = ((points_gpu[1, :] - self.y_range[0]) / self.resolution).long()
        
        # Clamp to image boundaries
        x_img = torch.clamp(x_img, 0, self.width - 1)
        y_img = torch.clamp(y_img, 0, self.height - 1)
        
        # Flip Y-axis
        y_img = self.height - 1 - y_img
        
        # Extract point attributes
        z_values = points_gpu[2, :]
        intensity_values = points_gpu[3, :]
        
        # Use scatter operations for parallel accumulation
        # For height: scatter_reduce with 'amax' (maximum)
        linear_indices = y_img * self.width + x_img
        height_flat = self.height_map.view(-1)
        
        # Ensure z_values has same dtype as height_flat
        z_values = z_values.float()
        
        # PyTorch doesn't have scatter_max with accumulation, so we use index_reduce
        # For each unique index, keep the maximum z value
        height_flat.scatter_reduce_(0, linear_indices, z_values, reduce='amax', include_self=False)
        
        # For intensity: sum then divide by count
        intensity_flat = self.intensity_sum.view(-1)
        intensity_values = intensity_values.float()
        intensity_flat.scatter_add_(0, linear_indices, intensity_values)
        
        # For density: count occurrences
        density_flat = self.density_map.view(-1)
        ones = torch.ones_like(linear_indices, dtype=torch.int32)
        density_flat.scatter_add_(0, linear_indices, ones)
        
        # Compute average intensity
        mask = self.density_map > 0
        intensity_map = torch.zeros_like(self.intensity_sum)
        intensity_map[mask] = self.intensity_sum[mask] / self.density_map[mask].float()
        
        # Normalize height map
        height_norm = torch.clamp(
            (self.height_map - self.z_range[0]) / (self.z_range[1] - self.z_range[0]),
            0, 1
        )
        height_norm = torch.pow(height_norm, 0.5)
        
        # Normalize intensity map
        intensity_max = intensity_map.max()
        if intensity_max > 1e-6:
            intensity_norm = intensity_map / intensity_max
        else:
            intensity_norm = intensity_map
        intensity_norm = torch.pow(intensity_norm, 0.5)
        
        # Normalize density map
        density_norm = torch.log1p(self.density_map.float())
        density_max = density_norm.max()
        if density_max > 1e-6:
            density_norm = density_norm / density_max
        density_norm = torch.pow(density_norm, 0.3)
        
        # Stack channels
        bev_image_gpu = torch.stack([height_norm, intensity_norm, density_norm], dim=-1)
        
        # Scale to uint8
        bev_image_gpu = (bev_image_gpu * 255).byte()
        
        # Transfer back to CPU as numpy array
        bev_image = bev_image_gpu.cpu().numpy()
        
        return bev_image
    
    def _rasterize_cpu(self, points):
        """
        CPU fallback implementation (identical to original BEVRasterizer).
        Used when GPU is not available.
        """
        # Initialize accumulation maps
        height_map = np.zeros((self.height, self.width), dtype=np.float32)
        intensity_map = np.zeros((self.height, self.width), dtype=np.float32)
        density_map = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Project 3D points to 2D pixel coordinates
        x_img = np.int32((points[0, :] - self.x_range[0]) / self.resolution)
        y_img = np.int32((points[1, :] - self.y_range[0]) / self.resolution)
        
        # Clamp to image boundaries
        x_img = np.clip(x_img, 0, self.width - 1)
        y_img = np.clip(y_img, 0, self.height - 1)
        
        # Flip Y-axis
        y_img = self.height - 1 - y_img
        
        # Accumulate values
        for i in range(points.shape[1]):
            x, y = x_img[i], y_img[i]
            height_map[y, x] = max(height_map[y, x], points[2, i])
            intensity_map[y, x] += points[3, i]
            density_map[y, x] += 1
        
        # Compute average intensity
        mask = density_map > 0
        intensity_map[mask] = intensity_map[mask] / density_map[mask]
        
        # Normalize height map
        height_map = np.clip(
            (height_map - self.z_range[0]) / (self.z_range[1] - self.z_range[0]),
            0, 1
        )
        height_map = np.power(height_map, 0.5)
        
        # Normalize intensity map
        intensity_map = intensity_map / max(intensity_map.max(), 1e-6)
        intensity_map = np.power(intensity_map, 0.5)
        
        # Normalize density map
        density_norm = np.log1p(density_map.astype(np.float32))
        density_norm = density_norm / max(density_norm.max(), 1e-6)
        density_norm = np.power(density_norm, 0.3)
        
        # Stack channels
        bev_image = np.stack([height_map, intensity_map, density_norm], axis=-1)
        
        # Scale to uint8
        return (bev_image * 255).astype(np.uint8)


def get_rasterizer(x_range=(-50, 50), y_range=(-50, 50), z_range=(-3, 5), resolution=0.1):
    """
    Factory function to create GPU-accelerated rasterizer.
    
    Args:
        x_range, y_range, z_range, resolution: Same as BEVRasterizer
    
    Returns:
        BEVRasterizerTorch instance
    """
    return BEVRasterizerTorch(x_range, y_range, z_range, resolution)


if __name__ == '__main__':
    """Test GPU rasterizer performance."""
    import time
    
    print("="*60)
    print("BEV Rasterizer GPU Performance Test (PyTorch)")
    print("="*60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ PyTorch CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("✗ CUDA not available")
    
    # Generate test point cloud
    print(f"\nGenerating test point cloud...")
    num_points = 100000
    points = np.random.randn(4, num_points).astype(np.float32)
    points[0, :] *= 40  # X: -40 to 40m
    points[1, :] *= 40  # Y: -40 to 40m
    points[2, :] = np.abs(points[2, :]) * 2  # Z: 0 to ~6m
    points[3, :] = np.random.rand(num_points) * 255  # Intensity: 0-255
    
    print(f"  Points: {num_points:,}")
    print(f"  Memory: {points.nbytes / 1024**2:.1f} MB")
    
    # Test GPU version
    print(f"\n{'='*60}")
    print("Testing GPU Rasterizer (1280x1280)")
    print(f"{'='*60}")
    
    rasterizer_gpu = BEVRasterizerTorch(resolution=0.078125)  # 1280x1280
    
    # Warmup
    _ = rasterizer_gpu.rasterize(points)
    if rasterizer_gpu.use_gpu:
        torch.cuda.synchronize()
    
    # Benchmark
    num_runs = 10
    start = time.time()
    for _ in range(num_runs):
        bev_gpu = rasterizer_gpu.rasterize(points)
        if rasterizer_gpu.use_gpu:
            torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = (time.time() - start) / num_runs
    
    print(f"  Image size: {bev_gpu.shape}")
    print(f"  Time per frame: {gpu_time*1000:.1f} ms")
    print(f"  FPS: {1/gpu_time:.1f}")
    print(f"  Mode: {'GPU' if rasterizer_gpu.use_gpu else 'CPU'}")
    
    # Test CPU version for comparison
    print(f"\n{'='*60}")
    print("Testing CPU Rasterizer (1280x1280)")
    print(f"{'='*60}")
    
    rasterizer_cpu = BEVRasterizerTorch(resolution=0.078125)
    rasterizer_cpu.use_gpu = False  # Force CPU mode
    
    start = time.time()
    bev_cpu = rasterizer_cpu.rasterize(points)
    cpu_time = time.time() - start
    
    print(f"  Image size: {bev_cpu.shape}")
    print(f"  Time per frame: {cpu_time*1000:.1f} ms")
    print(f"  FPS: {1/cpu_time:.1f}")
    
    # Speedup comparison
    if rasterizer_gpu.use_gpu:
        speedup = cpu_time / gpu_time
        print(f"\n{'='*60}")
        print(f"SPEEDUP: {speedup:.1f}x faster on GPU")
        print(f"{'='*60}")
        print(f"Performance gain: {(1-gpu_time/cpu_time)*100:.1f}%")
        print(f"Time saved: {(cpu_time - gpu_time)*1000:.1f} ms per frame")
        print(f"\nExpected total FPS with GPU BEV + YOLO:")
        print(f"  BEV: {gpu_time*1000:.0f}ms + YOLO: 40ms = {1/(gpu_time+0.040):.1f} FPS")
    else:
        print(f"\n{'='*60}")
        print("GPU not available - running in CPU mode")
        print(f"{'='*60}")
    
    print(f"\nTest complete!")
