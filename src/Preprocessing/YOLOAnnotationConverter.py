"""
YOLOAnnotationConverter: Transform 3D bounding boxes to YOLO format for BEV images.

This module converts nuScenes 3D bounding box annotations (in meters with global coordinates)
to YOLO-compatible 2D bounding box format (normalized pixel coordinates). It handles the
projection from 3D world space to 2D BEV image space and maps nuScenes' 23 object categories
to 4 simplified classes for object detection.

Key Operations:
- 3D bounding box → 2D bounding box projection (top-down view)
- Coordinate transformation: meters → pixels → normalized [0,1]
- Category mapping: 23 nuScenes classes → 4 detection classes
- Validation: Ensures boxes are within image bounds
"""

import numpy as np


class YOLOAnnotationConverter:
    """
    Converter for transforming 3D bounding boxes to YOLO format for BEV images.
    
    YOLO (You Only Look Once) format specifies 2D bounding boxes as:
    <class_id> <x_center> <y_center> <width> <height>
    
    where all spatial values are normalized to [0, 1] relative to image dimensions.
    
    This class performs the geometric transformation from 3D boxes in world coordinates
    to 2D boxes in BEV image coordinates, handling coordinate systems, resolution
    conversion, and class label mapping.
    
    Attributes:
        image_width: BEV image width in pixels
        image_height: BEV image height in pixels
        x_range: Spatial extent in X direction (meters)
        y_range: Spatial extent in Y direction (meters)
        resolution: Meters per pixel
        class_mapping: Dictionary mapping nuScenes categories to class IDs
    """
    
    def __init__(self, image_width, image_height, x_range=(-50, 50), y_range=(-50, 50), resolution=0.1):
        """
        Initialize the annotation converter with image and spatial parameters.
        
        Args:
            image_width: Width of BEV image in pixels (e.g., 1000)
            image_height: Height of BEV image in pixels (e.g., 1000)
            x_range: (min, max) spatial extent in X direction (meters)
            y_range: (min, max) spatial extent in Y direction (meters)
            resolution: Spatial resolution in meters per pixel (e.g., 0.1)
        
        Technical Details:
            - Image dimensions must match BEV rasterizer output
            - Spatial ranges must match point cloud filtering ranges
            - Resolution determines coordinate transformation accuracy
        """
        self.image_width = image_width
        self.image_height = image_height
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        # Map nuScenes' detailed taxonomy (23 classes) to simplified classes (4)
        # Rationale: Reduces class imbalance, focuses on key autonomous driving objects
        self.class_mapping = {
            # Class 0: Cars (most common, ~60% of objects)
            'vehicle.car': 0,
            'vehicle.taxi': 0,  # Taxis are functionally similar to cars
            
            # Class 1: Large Vehicles (trucks, buses, construction)
            'vehicle.truck': 1,
            'vehicle.bus.bendy': 1,  # Articulated buses
            'vehicle.bus.rigid': 1,   # Standard buses
            'vehicle.construction': 1,  # Bulldozers, cranes, etc.
            
            # Class 2: Pedestrians (all types, critical for safety)
            'human.pedestrian.adult': 2,
            'human.pedestrian.child': 2,
            'human.pedestrian.construction_worker': 2,
            'human.pedestrian.police_officer': 2,
            
            # Class 3: Two-wheeled Vehicles (vulnerable road users)
            'vehicle.bicycle': 3,
            'vehicle.motorcycle': 3
            
            # Note: Other nuScenes classes not included:
            # - vehicle.trailer, vehicle.emergency.*: Uncommon
            # - movable_object.*: Traffic cones, barriers (static)
            # - animal: Very rare in nuScenes dataset
        }
    
    def convert_annotation(self, box_translation, box_size, category_name):
        """
        Convert a single 3D bounding box annotation to YOLO format for BEV image.
        
        Transforms a 3D box defined in world coordinates (meters) to a 2D box in BEV
        image coordinates (normalized [0,1]). The conversion involves:
        1. Category filtering and mapping
        2. 3D→2D projection (top-down, uses only X and Y coordinates)
        3. Coordinate transformation (meters → pixels → normalized)
        4. Validation (ensure box is within image bounds)
        
        Args:
            box_translation: [x, y, z] center of 3D bounding box in ego frame (meters)
            box_size: [width, length, height] dimensions of 3D box (meters)
                     Note: nuScenes uses [width, length, height] order
            category_name: String category from nuScenes taxonomy (e.g., 'vehicle.car')
        
        Returns:
            List [class_id, x_center, y_center, width, height] in YOLO format, or None if:
            - Category is not in class_mapping (filtered out)
            - Box center is outside image bounds
            - Box dimensions are invalid (≤0 or >1 after normalization)
        
        Technical Details:
            
            **Coordinate System Transformation:**
            
            1. Input Space (3D world, ego vehicle frame):
               - Origin: Center of vehicle
               - X-axis: Forward (driving direction)
               - Y-axis: Left
               - Z-axis: Up (discarded in BEV)
               - Units: Meters
            
            2. Intermediate Space (2D pixel coordinates):
               - Origin: Top-left corner
               - X-axis: Right (horizontal)
               - Y-axis: Down (vertical)
               - Units: Pixels
               - Transformation: x_px = (x_m - x_min) / resolution
            
            3. Output Space (YOLO normalized coordinates):
               - Origin: Top-left corner
               - X-axis: Right, range [0, 1]
               - Y-axis: Down, range [0, 1]
               - Units: Fraction of image dimensions
               - Transformation: x_norm = x_px / image_width
            
            **Y-Axis Flip:**
            - World coordinates: Y increases leftward
            - Image coordinates: Y increases downward
            - Requires flip: y_img = image_height - 1 - y_world_to_px
            
            **Size Interpretation:**
            - box_size[0]: Width (lateral extent, X direction in world)
            - box_size[1]: Length (longitudinal extent, Y direction in world)
            - In BEV: Width maps to image X, Length maps to image Y
            - Z dimension (height) is discarded for 2D projection
            
            **Validation Logic:**
            - Filters unknown categories (returns None)
            - Rejects boxes with center outside [0, 1] range
            - Rejects boxes with invalid dimensions (too large or non-positive)
            - Clamps final values to [0, 1] as safety measure
        """
        # ============================================================
        # STEP 1: Filter by category (return None if not in mapping)
        # ============================================================
        if category_name not in self.class_mapping:
            return None  # Skip classes we're not detecting (e.g., trafficcone)
        
        class_id = self.class_mapping[category_name]  # Map to simplified class [0-3]
        
        # ============================================================
        # STEP 2: Convert 3D box center to 2D pixel coordinates
        # ============================================================
        
        # Transform X coordinate: world meters → pixel index
        # X in world frame (forward) → X in image (horizontal)
        x_center = (box_translation[0] - self.x_range[0]) / self.resolution
        
        # Transform Y coordinate: world meters → pixel index (before flip)
        y_center = (box_translation[1] - self.y_range[0]) / self.resolution
        
        # Flip Y-axis: image origin is top-left, world origin is center
        # In world: Y positive = left; In image: Y positive = down
        y_center = self.image_height - 1 - y_center
        
        # ============================================================
        # STEP 3: Convert 3D box size to 2D pixel dimensions
        # ============================================================
        
        # Width: box_size[0] is lateral extent (X direction)
        width = box_size[0] / self.resolution
        
        # Height: box_size[1] is longitudinal extent (Y direction)
        # Note: "height" in YOLO 2D means vertical extent in image, not Z
        height = box_size[1] / self.resolution
        
        # ============================================================
        # STEP 4: Normalize to [0, 1] range (YOLO format requirement)
        # ============================================================
        
        # Normalize center coordinates
        x_norm = x_center / self.image_width   # Fraction of image width
        y_norm = y_center / self.image_height  # Fraction of image height
        
        # Normalize dimensions
        w_norm = width / self.image_width
        h_norm = height / self.image_height
        
        # ============================================================
        # STEP 5: Validate box (reject invalid or out-of-bounds boxes)
        # ============================================================
        
        # Check if dimensions are valid (positive and not too large)
        if w_norm <= 0 or h_norm <= 0 or w_norm > 1 or h_norm > 1:
            return None  # Invalid box size
        
        # Check if center is within image bounds
        if x_norm < 0 or x_norm > 1 or y_norm < 0 or y_norm > 1:
            return None  # Box center outside image
        
        # ============================================================
        # STEP 6: Return YOLO format annotation
        # ============================================================
        
        # Format: [class_id, x_center, y_center, width, height]
        # All spatial values normalized to [0, 1]
        # Clamp as final safety check (should rarely trigger after validation)
        return [class_id, 
                np.clip(x_norm, 0, 1), 
                np.clip(y_norm, 0, 1), 
                np.clip(w_norm, 0, 1), 
                np.clip(h_norm, 0, 1)]

