"""
Example: Inference on Real LiDAR Data

This script demonstrates how to use the trained model for inference on real LiDAR scans.
It shows various use cases from single file inference to batch processing.
"""

from pathlib import Path
from src.Inference.LidarInference import LidarInference
import sys


def example_single_file_inference():
    """
    Example 1: Run inference on a single LiDAR file
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Single File Inference")
    print("="*70)
    
    # Path to your trained model
    model_path = 'build/runs/detect/stage2_finetune/weights/best.pt'
    
    # Find available LiDAR files
    lidar_dir = Path('build/data/raw/v1.0-trainval/samples/LIDAR_TOP')
    lidar_files = list(lidar_dir.glob('*.pcd.bin'))
    
    if not lidar_files:
        print(f"\n❌ No LiDAR files found in {lidar_dir}")
        return
    
    # Use first available file
    lidar_file = lidar_files[0]
    print(f"\nUsing LiDAR file: {lidar_file.name}")
    
    # Initialize inference engine
    inference = LidarInference(
        model_path=model_path,
        conf_threshold=0.3,  # Adjust confidence threshold
        iou_threshold=0.45
    )
    
    # Run inference
    results = inference.predict_from_lidar_file(lidar_file)
    
    # Print detections
    print(f"\nFound {len(results['detections'])} objects:")
    for i, det in enumerate(results['detections'], 1):
        print(f"{i}. {det['class_name']}: {det['confidence']:.2%}")
    
    # Visualize and save
    inference.visualize_detections(
        results,
        save_path='inference_output/single_detection.png'
    )
    
    print("\n✓ Complete! Check inference_output/single_detection.png")


def example_batch_processing():
    """
    Example 2: Process multiple LiDAR files in batch
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Processing")
    print("="*70)
    
    model_path = 'build/runs/detect/stage2_finetune/weights/best.pt'
    lidar_dir = Path('build/data/raw/v1.0-trainval/samples/LIDAR_TOP')
    output_dir = Path('inference_output/batch')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize once (reuse for all files)
    inference = LidarInference(model_path=model_path, conf_threshold=0.25)
    
    # Get first 5 LiDAR files for demo
    lidar_files = list(lidar_dir.glob('*.pcd.bin'))[:5]
    
    print(f"\nProcessing {len(lidar_files)} files...")
    
    all_detections = []
    for i, lidar_file in enumerate(lidar_files, 1):
        print(f"\n[{i}/{len(lidar_files)}] Processing {lidar_file.name}")
        
        # Run inference
        results = inference.predict_from_lidar_file(lidar_file)
        
        # Save visualization
        output_path = output_dir / f"{lidar_file.stem}_detected.png"
        inference.visualize_detections(results, save_path=output_path)
        
        # Collect statistics
        all_detections.extend(results['detections'])
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"  Total files processed: {len(lidar_files)}")
    print(f"  Total objects detected: {len(all_detections)}")
    
    # Count by class
    class_counts = {}
    for det in all_detections:
        cls = det['class_name']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print("\n  Detections by class:")
    for cls, count in sorted(class_counts.items()):
        print(f"    {cls}: {count}")
    
    print(f"\n✓ Complete! Check {output_dir}/ for visualizations")


def example_metric_coordinates():
    """
    Example 3: Get detection coordinates in meters (vehicle frame)
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Metric Coordinates (for downstream tasks)")
    print("="*70)
    
    model_path = 'build/runs/detect/stage2_finetune/weights/best.pt'
    
    # Find available LiDAR files
    lidar_dir = Path('build/data/raw/v1.0-trainval/samples/LIDAR_TOP')
    lidar_files = list(lidar_dir.glob('*.pcd.bin'))
    
    if not lidar_files:
        print(f"\n❌ No LiDAR files found in {lidar_dir}")
        return
    
    lidar_file = lidar_files[0]
    print(f"\nUsing LiDAR file: {lidar_file.name}")
    
    inference = LidarInference(model_path=model_path, conf_threshold=0.3)
    results = inference.predict_from_lidar_file(lidar_file)
    
    print("\nDetections in Vehicle Coordinate Frame:")
    print("(x: forward, y: left, origin at vehicle center)")
    print("-" * 70)
    
    for i, det in enumerate(results['detections'], 1):
        # Convert to metric coordinates
        det_meters = inference.get_detection_in_meters(det)
        
        x, y = det_meters['center']
        w, h = det_meters['size']
        
        print(f"\n{i}. {det_meters['class'].upper()}")
        print(f"   Confidence: {det_meters['confidence']:.2%}")
        print(f"   Position:   x={x:6.2f}m (forward), y={y:6.2f}m (left)")
        print(f"   Size:       {w:5.2f}m × {h:5.2f}m")
        
        # Calculate distance from vehicle
        distance = (x**2 + y**2)**0.5
        print(f"   Distance:   {distance:6.2f}m from vehicle")


def example_custom_visualization():
    """
    Example 4: Custom visualization with additional information
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Visualization")
    print("="*70)
    
    import cv2
    import numpy as np
    
    model_path = 'build/runs/detect/stage2_finetune/weights/best.pt'
    
    # Find available LiDAR files
    lidar_dir = Path('build/data/raw/v1.0-trainval/samples/LIDAR_TOP')
    lidar_files = list(lidar_dir.glob('*.pcd.bin'))
    
    if not lidar_files:
        print(f"\n❌ No LiDAR files found in {lidar_dir}")
        return
    
    lidar_file = lidar_files[0]
    print(f"\nUsing LiDAR file: {lidar_file.name}")
    
    inference = LidarInference(model_path=model_path, conf_threshold=0.25)
    results = inference.predict_from_lidar_file(lidar_file)
    
    # Get base visualization
    vis_image = inference.visualize_detections(results)
    
    # Add custom overlays
    # 1. Add grid lines (10m intervals)
    img_h, img_w = vis_image.shape[:2]
    resolution = inference.rasterizer.resolution
    meters_per_10px = 10 / resolution  # pixels for 10 meters
    
    # Draw vertical grid lines
    for i in range(0, img_w, int(meters_per_10px)):
        cv2.line(vis_image, (i, 0), (i, img_h), (100, 100, 100), 1)
    
    # Draw horizontal grid lines
    for i in range(0, img_h, int(meters_per_10px)):
        cv2.line(vis_image, (0, i), (img_w, i), (100, 100, 100), 1)
    
    # 2. Add distance circles (20m, 40m, 60m)
    center = (img_w // 2, img_h // 2)
    for radius_m in [20, 40, 60]:
        radius_px = int(radius_m / resolution)
        cv2.circle(vis_image, center, radius_px, (80, 80, 80), 1)
        # Label the circle
        label_pos = (center[0] + 5, center[1] - radius_px + 20)
        cv2.putText(vis_image, f"{radius_m}m", label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # 3. Add vehicle indicator at center
    cv2.circle(vis_image, center, 10, (0, 255, 255), -1)
    cv2.putText(vis_image, "EGO", (center[0] - 20, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Save custom visualization
    output_path = 'inference_output/custom_visualization.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, vis_image)
    
    print(f"\n✓ Custom visualization saved to: {output_path}")


def main():
    """
    Run all examples
    """
    print("\n" + "="*70)
    print("LiDAR Inference Examples")
    print("="*70)
    print("\nThese examples demonstrate how to use your trained model")
    print("for inference on real LiDAR data.")
    
    # Check if model exists
    model_path = Path('build/runs/detect/stage2_finetune/weights/best.pt')
    if not model_path.exists():
        print("\n❌ Error: Trained model not found!")
        print(f"   Expected location: {model_path}")
        print("\n   Please train the model first by running: python src/main.py")
        sys.exit(1)
    
    # Run examples
    try:
        example_single_file_inference()
        example_batch_processing()
        example_metric_coordinates()
        example_custom_visualization()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Check 'inference_output/' directory for visualizations")
        print("  2. Modify thresholds (conf_threshold, iou_threshold) as needed")
        print("  3. Process your own LiDAR files using the patterns above")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
