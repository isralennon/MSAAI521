# LiDAR Inference Guide

This guide shows you how to use your trained model to detect objects in real LiDAR data.

## Quick Start

### 1. Command-Line Inference (Single File)

```bash
# Basic inference on a single LiDAR file
python src/Inference/LidarInference.py \
    --model build/runs/detect/stage2_finetune/weights/best.pt \
    --lidar build/data/raw/v1.0-trainval/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151605547877.pcd.bin \
    --output my_detection.png

# Adjust confidence threshold
python src/Inference/LidarInference.py \
    --model build/runs/detect/stage2_finetune/weights/best.pt \
    --lidar path/to/your/lidar.pcd.bin \
    --output result.png \
    --conf 0.3 \
    --iou 0.45
```

### 2. Python Script Examples

Run the comprehensive examples:
```bash
python examples_inference.py
```

This will:
- Process single files
- Batch process multiple files
- Show metric coordinates
- Create custom visualizations

## Using the Inference API

### Basic Usage

```python
from src.Inference.LidarInference import LidarInference

# Initialize with your trained model
inference = LidarInference(
    model_path='build/runs/detect/stage2_finetune/weights/best.pt',
    conf_threshold=0.25,  # Confidence threshold (0-1)
    iou_threshold=0.45    # IoU threshold for NMS
)

# Run inference on a LiDAR file
results = inference.predict_from_lidar_file('path/to/lidar.pcd.bin')

# Visualize detections
inference.visualize_detections(results, save_path='output.png')
```

### Get Detection Information

```python
# Access detections
for det in results['detections']:
    print(f"Class: {det['class_name']}")
    print(f"Confidence: {det['confidence']:.2%}")
    print(f"Box (pixels): {det['box']}")  # [x1, y1, x2, y2]
```

### Convert to Metric Coordinates

```python
# Get position in meters (vehicle coordinate frame)
for det in results['detections']:
    det_meters = inference.get_detection_in_meters(det)
    
    x, y = det_meters['center']  # Position in meters
    w, h = det_meters['size']    # Size in meters
    
    print(f"{det_meters['class']}: x={x:.2f}m, y={y:.2f}m")
    print(f"  Size: {w:.2f}m × {h:.2f}m")
```

### Process Multiple Files

```python
from pathlib import Path

# Get all LiDAR files
lidar_dir = Path('build/data/raw/v1.0-trainval/samples/LIDAR_TOP')
lidar_files = list(lidar_dir.glob('*.pcd.bin'))

# Process each file
for lidar_file in lidar_files:
    results = inference.predict_from_lidar_file(lidar_file)
    
    # Save visualization
    output_name = f"{lidar_file.stem}_detected.png"
    inference.visualize_detections(results, save_path=f"outputs/{output_name}")
    
    print(f"Processed {lidar_file.name}: {len(results['detections'])} detections")
```

## Understanding the Output

### Coordinate Systems

**Pixel Coordinates (BEV Image)**
- Origin: Top-left corner
- X-axis: Right (0 to 1024)
- Y-axis: Down (0 to 1024)

**Metric Coordinates (Vehicle Frame)**
- Origin: Vehicle center
- X-axis: Forward (front of vehicle)
- Y-axis: Left
- Z-axis: Up
- Unit: Meters

### Detection Dictionary Structure

```python
detection = {
    'box': [x1, y1, x2, y2],      # Bounding box in pixels
    'class_id': 0,                 # Class ID (0-3)
    'class_name': 'car',           # Class name
    'confidence': 0.85             # Detection confidence (0-1)
}
```

### Class Mapping

- **0**: Car (Green)
- **1**: Truck/Bus (Blue)
- **2**: Pedestrian (Red)
- **3**: Cyclist (Cyan)

## Advanced Usage

### Custom Preprocessing

```python
# Load LiDAR manually
points = inference.load_lidar_from_file('lidar.pcd.bin')

# Custom filtering (if needed)
# points = your_custom_filter(points)

# Preprocess to BEV
bev_image = inference.preprocess_lidar(points)

# Run detection
results = inference.predict(bev_image)
```

### Integration with Robotics Stack

```python
import rospy
from sensor_msgs.msg import PointCloud2

def lidar_callback(msg):
    # Convert ROS PointCloud2 to numpy array
    points = pointcloud2_to_array(msg)  # Your conversion function
    
    # Preprocess and detect
    bev_image = inference.preprocess_lidar(points)
    results = inference.predict(bev_image)
    
    # Publish detections
    for det in results['detections']:
        det_meters = inference.get_detection_in_meters(det)
        # Publish to /detections topic
        publish_detection(det_meters)
```

### Custom Visualization

```python
import cv2

# Get base visualization
results = inference.predict_from_lidar_file('lidar.pcd.bin')
vis_image = inference.visualize_detections(results)

# Add custom overlays
# Example: Draw distance circles
center = (512, 512)  # Image center for 1024x1024
for radius_m in [20, 40, 60]:
    radius_px = int(radius_m / inference.rasterizer.resolution)
    cv2.circle(vis_image, center, radius_px, (100, 100, 100), 1)

# Save
cv2.imwrite('custom_output.png', vis_image)
```

## Tuning Parameters

### Confidence Threshold
- **Default**: 0.25
- **Lower** (e.g., 0.15): More detections, more false positives
- **Higher** (e.g., 0.4): Fewer detections, higher precision

### IoU Threshold
- **Default**: 0.45
- **Lower** (e.g., 0.3): More aggressive NMS, fewer overlapping boxes
- **Higher** (e.g., 0.6): Keep more overlapping detections

```python
# Experiment with thresholds
inference = LidarInference(
    model_path='model.pt',
    conf_threshold=0.3,  # Adjust based on your needs
    iou_threshold=0.5
)
```

## Troubleshooting

### Issue: No detections found
**Solutions:**
1. Lower confidence threshold: `conf_threshold=0.15`
2. Check if LiDAR file has data: `points.shape`
3. Visualize BEV image to ensure proper preprocessing

### Issue: Too many false positives
**Solutions:**
1. Increase confidence threshold: `conf_threshold=0.35`
2. Use the best weights from stage 2 fine-tuning
3. Check if model was trained on similar data

### Issue: Detections in wrong location
**Solutions:**
1. Verify coordinate frame conversion
2. Check BEV resolution matches training (1024x1024)
3. Ensure LiDAR calibration is correct

## Performance Optimization

### Batch Processing
```python
# Process multiple files efficiently
inference = LidarInference(model_path='model.pt')

for lidar_file in lidar_files:
    # Reuse the same inference object
    results = inference.predict_from_lidar_file(lidar_file)
```

### GPU Acceleration
The model automatically uses GPU if available. Check with:
```python
import torch
print(f"Using GPU: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## Example Output

```
Loading model from: build/runs/detect/stage2_finetune/weights/best.pt
✓ Model loaded successfully
  Confidence threshold: 0.25
  IoU threshold: 0.45
  BEV resolution: 0.097656 m/pixel
  Image size: 1024x1024

Running inference on: lidar_scan.pcd.bin
Loaded 34720 points from lidar_scan.pcd.bin
Generated BEV image: (1024, 1024, 3)
Detected 12 objects

==============================================================
DETECTIONS (Vehicle Frame - Meters)
==============================================================

1. CAR (conf: 0.87)
   Center: x= 15.23m, y= -3.45m
   Size:   w=  4.12m, h=  1.89m

2. PEDESTRIAN (conf: 0.72)
   Center: x=  8.91m, y=  2.15m
   Size:   w=  0.65m, h=  0.71m

✓ Saved visualization to: output.png
```

## Next Steps

1. **Test on your data**: Replace LiDAR file paths with your own data
2. **Tune thresholds**: Experiment with conf/IoU thresholds for your use case
3. **Integrate**: Use the API in your autonomous driving stack
4. **Extend**: Add tracking, path prediction, or sensor fusion

## Additional Resources

- Model training: `src/main.py`
- Evaluation metrics: `src/Evaluation/ModelEvaluator.py`
- BEV visualization: `src/Preprocessing/BEVInspector.py`
