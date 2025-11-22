"""
DataPreprocessor: Orchestrate the full nuScenes → YOLO BEV dataset conversion pipeline.

This is the main preprocessing orchestrator that combines all preprocessing components to
convert the raw nuScenes dataset into a YOLO-compatible BEV object detection dataset.

Pipeline Overview:
1. Load 3D LiDAR point cloud → PointCloudProcessor
2. Transform to ego vehicle frame → PointCloudProcessor
3. Filter to region of interest → PointCloudProcessor
4. Rasterize to BEV image → BEVRasterizer
5. Transform 3D annotations to 2D YOLO format → YOLOAnnotationConverter
6. Save paired images and labels to disk

Output Structure:
    build/data/preprocessed/
    ├── images/
    │   ├── scene-0001_<token>.png
    │   ├── scene-0002_<token>.png
    │   └── ...
    └── labels/
        ├── scene-0001_<token>.txt
        ├── scene-0002_<token>.txt
        └── ...
"""

from Preprocessing.PointCloudProcessor import PointCloudProcessor
from Preprocessing.BEVRasterizer import BEVRasterizer
from Preprocessing.YOLOAnnotationConverter import YOLOAnnotationConverter
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import numpy as np
import cv2
import os
from pathlib import Path
from Globals import PREPROCESSED_ROOT


class DataPreprocessor:
    """
    Main preprocessing pipeline orchestrator for nuScenes → YOLO BEV conversion.
    
    This class coordinates the entire preprocessing workflow, managing the flow of data
    through each processing stage and handling file I/O for the output dataset.
    
    The preprocessor creates a dataset suitable for training YOLO models on BEV images,
    with each sample consisting of:
    - A 3-channel BEV image (PNG format, 1000×1000 pixels by default)
    - A corresponding YOLO format label file (TXT format, one box per line)
    
    Attributes:
        nusc: NuScenes dataset instance
        pc_processor: PointCloudProcessor for loading and filtering point clouds
        rasterizer: BEVRasterizer for converting point clouds to images
        converter: YOLOAnnotationConverter for transforming annotations
        output_root: Root directory for preprocessed dataset
        images_dir: Directory for BEV images
        labels_dir: Directory for YOLO labels
    """
    
    def __init__(self, nusc):
        """
        Initialize the data preprocessor with all required components.
        
        Args:
            nusc: NuScenes instance providing access to the raw dataset
        
        Technical Details:
            - Creates instances of all preprocessing components with compatible parameters
            - Sets up output directory structure (creates if doesn't exist)
            - Ensures image dimensions match between rasterizer and annotation converter
        """
        self.nusc = nusc
        
        # Initialize preprocessing components with default parameters
        self.pc_processor = PointCloudProcessor(nusc)
        self.rasterizer = BEVRasterizer()
        
        # Pass rasterizer dimensions to ensure coordinate transformation consistency
        self.converter = YOLOAnnotationConverter(
            self.rasterizer.width, 
            self.rasterizer.height
        )
        
        # Setup output directory structure
        self.output_root = Path(PREPROCESSED_ROOT)
        self.images_dir = self.output_root / 'images'  # BEV PNG files
        self.labels_dir = self.output_root / 'labels'  # YOLO TXT files
        
        # Create directories (parents=True creates intermediate dirs, exist_ok ignores if exists)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
    
    def process_all_samples(self):
        """
        Process all samples in the nuScenes dataset and save as YOLO BEV dataset.
        
        This is the main processing loop that iterates through every sample (timestamp)
        in the nuScenes dataset, generates a BEV image and YOLO annotations, and saves
        them to disk in a format suitable for training object detection models.
        
        Returns:
            int: Total number of samples processed
        
        Processing Pipeline per Sample:
            1. Load LiDAR point cloud
            2. Transform to ego vehicle frame
            3. Filter to region of interest
            4. Rasterize to BEV image
            5. Transform all 3D annotations to 2D YOLO format
            6. Save image and labels with matching filenames
        
        Technical Details:
            
            **Coordinate Frame Transformations:**
            The preprocessing involves two key coordinate transformations:
            
            a) Point Cloud Transformation (handled by PointCloudProcessor):
               - Sensor frame → Ego vehicle frame
               - Applied via calibrated_sensor record (rotation + translation)
            
            b) Annotation Transformation (handled in this method):
               - Global frame → Ego vehicle frame
               - Required because annotations are in global coordinates
               - Uses ego_pose to transform: T_ego^global
               - Inverse transformation: box_ego = T_ego^global^-1 * box_global
            
            **Why Two Transformations?**
            - Point clouds: Stored in sensor frame, need ego frame alignment
            - Annotations: Stored in global frame, need ego frame alignment
            - Both must be in same frame (ego) for consistent BEV projection
            
            **File Naming Convention:**
            - Format: {scene_name}_{sample_token}.{ext}
            - Example: scene-0061_3e8750f331d7499e9b5123e9eb70f2e2.png
            - Ensures unique names and preserves scene context
            - Matching names for image-label pairs enable automatic pairing
            
            **YOLO Label Format (per line in .txt file):**
            <class_id> <x_center> <y_center> <width> <height>
            - class_id: Integer [0-3]
            - All other values: Floats in range [0.0, 1.0]
            - Space-separated values
            - One line per object
            
            **Performance Considerations:**
            - v1.0-mini: ~400 samples, takes ~5-10 minutes to process
            - Full dataset: ~40,000 samples, takes hours
            - Each BEV image: ~3MB (1000×1000×3)
            - Total output: ~1.2GB for mini, ~120GB for full dataset
        """
        # Get total number of samples to process (v1.0-mini has ~400)
        total_samples = len(self.nusc.sample)
        
        # ============================================================
        # Main Processing Loop: Iterate through all samples
        # ============================================================
        for sample_idx in range(total_samples):
            # Get the sample record (represents one timestamp across all sensors)
            sample = self.nusc.sample[sample_idx]
            
            # Extract LIDAR_TOP token (primary 3D sensor for BEV generation)
            lidar_token = sample['data']['LIDAR_TOP']
            
            # --------------------------------------------------------
            # STAGE 1: Generate BEV Image from Point Cloud
            # --------------------------------------------------------
            
            # Load and transform point cloud to ego vehicle frame
            points = self.pc_processor.load_and_transform(lidar_token)
            
            # Filter to region of interest (removes distant/irrelevant points)
            filtered_points = self.pc_processor.filter_points(points)
            
            # Rasterize 3D points to 2D BEV image (height, intensity, density channels)
            bev_image = self.rasterizer.rasterize(filtered_points)
            
            # --------------------------------------------------------
            # STAGE 2: Transform Annotations to YOLO Format
            # --------------------------------------------------------
            
            # ============================================================
            # COORDINATE FRAME TRANSFORMATION: Global → Ego → Sensor
            # ============================================================
            # PROBLEM: nuScenes stores data in different coordinate frames:
            # - LiDAR point clouds: Sensor frame (LIDAR_TOP coordinate system)
            # - Annotations (bounding boxes): Global frame (world coordinates)
            # 
            # SOLUTION: Transform annotations to match LiDAR's sensor frame
            # Transformation chain: Global → Ego Vehicle → Sensor
            # 
            # WHY THIS IS NECESSARY:
            # For proper BEV projection, both point cloud and annotations must be
            # in the same coordinate system. Since we keep LiDAR in sensor frame
            # (see PointCloudProcessor.py), we must transform annotations TO sensor frame.
            # ============================================================
            
            # Get ego vehicle pose for this timestamp (needed for global→ego transform)
            sample_data = self.nusc.get('sample_data', lidar_token)
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            
            # Get sensor calibration for ego→sensor transformation
            cs_record = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            
            # Process all annotations (3D bounding boxes) for this sample
            yolo_labels = []
            for ann_token in sample['anns']:
                # Get annotation metadata (stored in global/world coordinates)
                ann = self.nusc.get('sample_annotation', ann_token)
                
                # Create 3D box object from annotation (currently in global frame)
                box = Box(
                    ann['translation'],  # [x, y, z] center in global frame
                    ann['size'],         # [width, length, height] dimensions
                    Quaternion(ann['rotation'])  # Orientation quaternion
                )
                
                # --------------------------------------------------------
                # TRANSFORMATION STEP 1 & 2: Global Frame → Ego Vehicle Frame
                # --------------------------------------------------------
                # The ego vehicle frame has its origin at the center of the vehicle
                # with X=forward, Y=left, Z=up (right-handed system)
                
                # Step 1: Translate by negative ego position (center on ego)
                box.translate(-np.array(ego_pose['translation']))
                
                # Step 2: Rotate by inverse ego orientation (align with ego axes)
                box.rotate(Quaternion(ego_pose['rotation']).inverse)
                
                # --------------------------------------------------------
                # TRANSFORMATION STEP 3 & 4: Ego Frame → Sensor Frame
                # --------------------------------------------------------
                # CRITICAL FIX: This transformation was missing in the original code,
                # causing misalignment between LiDAR points and bounding boxes.
                # 
                # The sensor frame (LIDAR_TOP) has a different origin and orientation
                # than the ego frame, so we must apply the sensor calibration transform.
                
                # Step 3: Translate by negative sensor position (center on sensor)
                box.translate(-np.array(cs_record['translation']))
                
                # Step 4: Rotate by inverse sensor orientation (align with sensor axes)
                box.rotate(Quaternion(cs_record['rotation']).inverse)
                
                # --------------------------------------------------------
                # AXIS-ALIGNED BOUNDING BOX COMPUTATION
                # --------------------------------------------------------
                # PROBLEM: Standard YOLO format only supports axis-aligned bounding boxes
                # (no rotation angle). However, our 3D boxes are rotated in 3D space.
                # 
                # SOLUTION: Compute the minimum axis-aligned bounding box (AABB) that
                # fully contains the rotated 3D box when viewed from above (BEV).
                # 
                # WHY THIS IS NECESSARY:
                # - A car at 45° has a larger footprint in axis-aligned coordinates
                # - Using original width/length would create boxes that don't fully
                #   contain the rotated object
                # - AABB ensures the box properly encloses the object at any angle
                # 
                # ALGORITHM:
                # 1. Get all 8 corners of the rotated 3D box
                # 2. Find min/max X and Y coordinates (top-down projection)
                # 3. Compute new center and dimensions from these extents
                # --------------------------------------------------------
                
                # Get all 8 corners of the rotated 3D box (3×8 array: x, y, z for each corner)
                corners = box.corners()
                
                # For BEV (top-down view), we only need X and Y coordinates (Z is discarded)
                # Find the min/max extents in X and Y to create the smallest axis-aligned box
                x_min, x_max = corners[0, :].min(), corners[0, :].max()
                y_min, y_max = corners[1, :].min(), corners[1, :].max()
                
                # Compute axis-aligned center (midpoint of extents)
                # Z coordinate remains unchanged (height doesn't affect top-down projection)
                aa_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, box.center[2]])
                
                # Compute axis-aligned dimensions (extent ranges)
                # Width = X extent, Length = Y extent, Height = original Z dimension
                aa_size = np.array([x_max - x_min, y_max - y_min, box.wlh[2]])
                
                # Convert 3D axis-aligned box (sensor frame) to 2D YOLO annotation (BEV image)
                yolo_label = self.converter.convert_annotation(
                    aa_center,          # Axis-aligned box center in sensor frame
                    aa_size,            # Axis-aligned box dimensions [width, length, height]
                    ann['category_name']  # Object category for class mapping
                )
                
                # Add to list if valid (converter returns None for invalid/filtered boxes)
                if yolo_label:
                    yolo_labels.append(yolo_label)
            
            # --------------------------------------------------------
            # STAGE 3: Save Image and Labels to Disk
            # --------------------------------------------------------
            
            # Generate unique filename from scene name and sample token
            scene_name = self.nusc.get('scene', sample['scene_token'])['name']
            filename = f"{scene_name}_{sample['token']}"
            
            # Save BEV image as PNG (3-channel RGB, uint8)
            image_path = self.images_dir / f"{filename}.png"
            cv2.imwrite(str(image_path), bev_image)
            
            # Save YOLO labels as TXT (one line per object)
            label_path = self.labels_dir / f"{filename}.txt"
            with open(label_path, 'w') as f:
                for label in yolo_labels:
                    # Format: <class> <x> <y> <w> <h>
                    f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")
        
        # Return total number of processed samples
        return total_samples

