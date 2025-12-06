# Inference and Visualization Guide

This guide covers the inference capabilities and real-time visualization tools added to the nuScenes BEV object detection project.

## Overview

The project now includes:
1. **LiDAR Inference Engine** - Run trained models on LiDAR data
2. **2D Real-time Visualizer** - Bird's eye view playback (like the training data)
3. **3D Real-time Visualizer** - Interactive 3D point cloud visualization with bounding boxes

## Quick Start

### 1. Basic Inference

Run inference on a single LiDAR file:

```python
from src.Inference.LidarInference import LidarInference

# Initialize inference engine
inference = LidarInference(
    model_path='build/runs/detect/stage2_finetune/weights/best.pt',
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Run inference
results = inference.predict_from_lidar_file('path/to/lidar.pcd.bin')

# Access detections
for det in results['detections']:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
```

### 2. Real-time 2D Visualization (Bird's Eye View)

Visualize detections in 2D BEV format (matching training data):

```bash
python realtime_visualizer_2d.py \
  --source build/data/raw/v1.0-trainval/samples/LIDAR_TOP \
  --model build/runs/detect/stage2_finetune/weights/best.pt
```

**Controls:**
- `SPACE` - Pause/Resume
- `Q` - Quit
- `+/-` - Speed up/down
- `N` - Next frame (when paused)
- `R` - Reset to start
- `S` - Save current frame

### 3. Real-time 3D Visualization (Point Cloud)

Visualize detections in 3D with interactive controls:

```bash
python realtime_visualizer_3d_simple.py \
  --source build/data/raw/v1.0-trainval/samples/LIDAR_TOP \
  --model build/runs/detect/stage2_finetune/weights/best.pt \
  --max-frames 100
```

**Controls:**
- `SPACE` - Pause/Resume
- `Q` - Quit
- `+/-` - Speed up/down
- `N` - Next frame (when paused)
- `R` - Reset to start
- `Mouse Scroll` - Zoom in/out
- `Mouse Drag` - Rotate view

**Features:**
- Blue point cloud (matching 2D visualizer style)
- Distance rings at 10m, 20m, 30m, 40m
- 3D bounding boxes with class colors (red=car, blue=truck/bus, green=pedestrian, yellow=cyclist)
- Real-time FPS and detection stats
- Dark background for better visibility

## Inference Module Documentation

### LidarInference Class

Location: `src/Inference/LidarInference.py`

#### Methods

**`__init__(model_path, conf_threshold=0.25, iou_threshold=0.45)`**
- Initialize inference engine with trained model
- Args:
  - `model_path`: Path to trained YOLO .pt file
  - `conf_threshold`: Confidence threshold for detections (0-1)
  - `iou_threshold`: IoU threshold for NMS (0-1)

**`load_lidar_from_file(lidar_file_path)`**
- Load LiDAR point cloud from .pcd.bin file
- Returns: Numpy array (4, N) with [x, y, z, intensity]

**`preprocess_lidar(points)`**
- Convert point cloud to BEV image
- Returns: 1024x1024x3 BGR image

**`predict(bev_image)`**
- Run YOLO model on BEV image
- Returns: YOLO detection results

**`predict_from_lidar_file(lidar_file_path)`**
- End-to-end inference from LiDAR file
- Returns: Dictionary with:
  - `bev_image`: Generated BEV image
  - `results`: YOLO detection results
  - `detections`: List of detection dictionaries

**Detection Dictionary Format:**
```python
{
    'box': [x1, y1, x2, y2],      # Pixel coordinates (xyxy format)
    'class_id': int,               # 0=car, 1=truck_bus, 2=pedestrian, 3=cyclist
    'class_name': str,             # Human-readable class name
    'confidence': float            # Confidence score (0-1)
}
```

## Dataset Compatibility

The code automatically detects which nuScenes version is available:

1. **v1.0-trainval** (full dataset) - Used if available
2. **v1.0-mini** (sample dataset) - Used as fallback

This is handled in `src/Globals.py` which auto-detects the dataset version in `build/data/raw/`.

## Training Pipeline

The branch includes a complete training pipeline:

```bash
# Full pipeline (download, preprocess, train, evaluate)
python src/main.py
```

**Pipeline stages:**
1. Data download and validation
2. Preprocessing (LiDAR → BEV conversion)
3. Dataset splitting (70/15/15)
4. Two-stage training:
   - Stage 1: Warmup (50 epochs)
   - Stage 2: Fine-tuning (150 epochs)
5. Evaluation and visualization

## Model Performance

On v1.0-trainval (3376 samples):
- **mAP@0.5**: 0.6163
- **mAP@0.5:0.95**: 0.3786
- **Classes**: car, truck_bus, pedestrian, cyclist

## File Structure

```
MSAAI521/
├── src/
│   ├── Inference/
│   │   └── LidarInference.py         # Inference engine
│   ├── Training/
│   │   ├── ModelInitializer.py       # Model setup
│   │   └── TrainingOrchestrator.py   # Training pipeline
│   ├── Evaluation/
│   │   ├── ModelEvaluator.py         # Performance metrics
│   │   ├── PerformanceAnalyzer.py    # Analysis tools
│   │   └── ResultsVisualizer.py      # Visualization
│   └── DataPreparation/
│       ├── DataSplitter.py           # Train/val/test split
│       └── DatasetConfigGenerator.py # YOLO config
├── realtime_visualizer_2d.py         # 2D BEV visualizer
├── realtime_visualizer_3d_simple.py  # 3D point cloud visualizer
└── README_INFERENCE.md               # This file
```

## Requirements

```
ultralytics>=8.0.0
opencv-python
numpy
matplotlib
nuscenes-devkit
```

## Tips

1. **Performance**: The 3D visualizer subsamples points for performance. Adjust subsample rate in code if needed.
2. **Zoom**: Use mouse scroll in 3D view to zoom in on specific areas
3. **Recording**: 2D visualizer supports video recording with `--record output.mp4`
4. **Speed**: Adjust playback speed with +/- keys
5. **Dataset**: Works with both v1.0-mini and v1.0-trainval automatically

## Troubleshooting

**Issue**: "No module named 'src.Inference'"
- **Fix**: Run from project root directory

**Issue**: 3D visualizer shows black screen
- **Fix**: Check that matplotlib backend is properly set (should use TkAgg)

**Issue**: Bounding boxes misaligned
- **Fix**: Ensure using correct model resolution (1024x1024)

**Issue**: ImportError for tkinter
- **Fix**: Install with `sudo apt-get install python3-tk`

## Citation

If using this code, please cite the nuScenes dataset:
```
@inproceedings{nuscenes2019,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H and Vora, Sourabh and 
          Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and 
          Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={CVPR},
  year={2020}
}
```
