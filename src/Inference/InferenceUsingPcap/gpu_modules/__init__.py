"""
GPU-accelerated modules for LiDAR inference

This package contains GPU-accelerated implementations of computationally
intensive operations used in LiDAR-based object detection.

Modules:
--------
- bev_rasterizer_torch.py: PyTorch-based GPU BEV rasterization (11.7x speedup)
- bev_rasterizer_gpu.py: CuPy-based GPU BEV rasterization (alternative)

Usage:
------
The gpu_modules are automatically imported by lidar_inference.py when available.
No manual import needed for standard usage.

Performance:
-----------
CPU (NumPy): ~800ms per frame, 1-2 FPS
GPU (PyTorch): ~15ms per frame, 13-17 FPS end-to-end

Requirements:
------------
- PyTorch with CUDA support (automatically detected)
- NVIDIA GPU with CUDA capability
"""

__version__ = "1.0.0"
__all__ = ["BEVRasterizerTorch", "BEVRasterizerGPU"]

from .bev_rasterizer_torch import BEVRasterizerTorch

try:
    from .bev_rasterizer_gpu import BEVRasterizerGPU
except ImportError:
    BEVRasterizerGPU = None
