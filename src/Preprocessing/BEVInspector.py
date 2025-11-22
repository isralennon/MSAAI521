"""
BEVInspector: Visualize and validate preprocessed BEV dataset with YOLO annotations.

This module provides tools for inspecting the preprocessed BEV dataset by loading
BEV images and their corresponding YOLO annotations, then visualizing the bounding
boxes overlaid on the images. This is essential for:
- Validating preprocessing pipeline correctness
- Debugging annotation alignment issues
- Visual quality assessment of the dataset
- Understanding class distribution and box sizes
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import os
from Globals import PREPROCESSED_ROOT


class BEVInspector:
    """
    Inspector for visualizing preprocessed BEV images with YOLO bounding boxes.
    
    This class loads pairs of BEV images and their corresponding YOLO label files,
    renders the bounding boxes on the images with color-coded class labels, and
    displays them for visual inspection and quality control.
    
    Visualization helps identify issues such as:
    - Misaligned bounding boxes (coordinate transformation errors)
    - Missing or extra annotations
    - Incorrect class labels
    - Box size anomalies
    
    Attributes:
        class_names: List of human-readable class names [Car, Truck/Bus, Pedestrian, Cyclist]
        colors: List of BGR color tuples for each class (for OpenCV rendering)
        images_dir: Path to directory containing BEV images
        labels_dir: Path to directory containing YOLO label files
    """
    
    def __init__(self):
        """
        Initialize the BEV inspector with class definitions and data paths.
        
        Technical Details:
            - Class names map to class IDs [0, 1, 2, 3]
            - Colors are in BGR format (OpenCV convention, not RGB)
            - Color scheme:
              * Red (255,0,0): Cars - most common, high visibility
              * Green (0,255,0): Trucks/Buses - large vehicles
              * Blue (0,0,255): Pedestrians - vulnerable road users
              * Yellow (255,255,0): Cyclists - two-wheeled vehicles
        """
        # Human-readable class labels (index = class ID)
        self.class_names = ['Car', 'Truck/Bus', 'Pedestrian', 'Cyclist']
        
        # Colors for bounding boxes (BGR format for OpenCV)
        self.colors = [
            (255, 0, 0),   # Class 0 (Car): Red
            (0, 255, 0),   # Class 1 (Truck/Bus): Green
            (0, 0, 255),   # Class 2 (Pedestrian): Blue
            (255, 255, 0)  # Class 3 (Cyclist): Yellow
        ]
        
        # Paths to preprocessed dataset directories
        self.images_dir = Path(PREPROCESSED_ROOT) / 'images'
        self.labels_dir = Path(PREPROCESSED_ROOT) / 'labels'
    
    def load_samples(self, num_samples):
        """
        Load a subset of BEV images and their corresponding YOLO labels.
        
        This method samples images uniformly across the dataset to provide a
        representative view of the preprocessed data without loading everything.
        
        Args:
            num_samples: Number of image-label pairs to load
        
        Returns:
            Tuple of (bev_images, yolo_labels_list) where:
            - bev_images: List of numpy arrays (BGR images from OpenCV)
            - yolo_labels_list: List of lists, each containing YOLO annotations
              Format per annotation: [class_id, x_center, y_center, width, height]
        
        Technical Details:
            1. Sampling Strategy:
               - Lists all PNG files in images directory
               - Calculates uniform step size to select evenly distributed samples
               - Avoids random sampling to ensure reproducibility
            
            2. File Loading:
               - Images: cv2.imread loads as BGR uint8 arrays
               - Labels: Text file parsed line-by-line
               - Matching: Uses image filename stem to find corresponding label
            
            3. Label Parsing:
               - Splits each line by whitespace
               - Expects exactly 5 values per line
               - First value (class_id) converted to int
               - Remaining values (x, y, w, h) converted to float
               - Invalid lines (wrong format) are skipped
            
            4. Error Handling:
               - Missing label files result in empty label list (not an error)
               - Malformed lines within label files are silently skipped
               - This gracefully handles incomplete preprocessing
        """
        # Get all BEV image files, sorted for consistent ordering
        image_files = sorted(list(self.images_dir.glob('*.png')))
        
        # Calculate step size for uniform sampling across dataset
        step = max(1, len(image_files) // num_samples)
        
        # Select evenly-spaced images (list comprehension with stride)
        selected_files = [image_files[i * step] for i in range(num_samples) if i * step < len(image_files)]
        
        # Initialize lists to collect loaded data
        bev_images = []
        yolo_labels_list = []
        
        # Load each selected image and its corresponding labels
        for image_path in selected_files:
            # Load BEV image (BGR format, uint8)
            bev_image = cv2.imread(str(image_path))
            
            # Find matching label file (same filename, different extension)
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            yolo_labels = []
            
            # Parse YOLO labels if file exists
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()  # Split by whitespace
                        
                        # Validate format: must have exactly 5 values
                        if len(parts) == 5:
                            # Parse: <class_id> <x> <y> <w> <h>
                            yolo_labels.append(
                                [int(parts[0])] +  # class_id as integer
                                [float(x) for x in parts[1:]]  # coordinates as floats
                            )
            
            # Add to collections
            bev_images.append(bev_image)
            yolo_labels_list.append(yolo_labels)
        
        return bev_images, yolo_labels_list
    
    def visualize(self, bev_image, yolo_labels):
        """
        Visualize a single BEV image with bounding boxes and labels overlaid.
        
        Creates a matplotlib figure showing the BEV image with color-coded bounding
        boxes drawn for each object, along with class name labels.
        
        Args:
            bev_image: numpy array (H, W, 3) in BGR format from OpenCV
            yolo_labels: List of annotations, each [class_id, x, y, w, h] normalized
        
        Technical Details:
            1. Color Space Conversion:
               - Input: BGR (OpenCV format)
               - Output: RGB (matplotlib format)
               - Required because OpenCV and matplotlib use different conventions
            
            2. Coordinate Denormalization:
               - YOLO format uses normalized coordinates [0, 1]
               - Must multiply by image dimensions to get pixel coordinates
               - x_pixel = x_normalized * image_width
            
            3. Box Format Conversion:
               - YOLO: (center_x, center_y, width, height)
               - OpenCV rectangle: (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
               - Conversion: top_left = center - size/2, bottom_right = center + size/2
            
            4. Rendering:
               - Boxes drawn with 2-pixel thickness
               - Class names positioned above top-left corner
               - Colors match class definitions
        """
        # Convert BGR (OpenCV) to RGB (matplotlib)
        img_rgb = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]  # Get image dimensions
        
        # Draw each bounding box and label
        for label in yolo_labels:
            class_id = int(label[0])  # Object class [0-3]
            
            # Denormalize coordinates: [0,1] → pixel values
            x_center, y_center = label[1] * w, label[2] * h
            box_w, box_h = label[3] * w, label[4] * h
            
            # Convert from center-size to corner format
            x1 = int(x_center - box_w / 2)  # Top-left X
            y1 = int(y_center - box_h / 2)  # Top-left Y
            x2 = int(x_center + box_w / 2)  # Bottom-right X
            y2 = int(y_center + box_h / 2)  # Bottom-right Y
            
            # Draw rectangle (modifies image in-place)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), self.colors[class_id], 2)
            
            # Draw class label above box
            cv2.putText(img_rgb, self.class_names[class_id], (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[class_id], 2)
        
        # Display using matplotlib
        plt.figure(figsize=(12, 12))
        plt.imshow(img_rgb)
        plt.title('BEV Image with YOLO Annotations')
        plt.axis('off')  # Hide axis ticks and labels
        plt.show()
    
    def visualize_grid(self, bev_images, yolo_labels_list, num_cols=2):
        """
        Visualize multiple BEV images in a grid layout.
        
        Public interface method that delegates to _draw_grid for rendering.
        
        Args:
            bev_images: List of BEV images (numpy arrays)
            yolo_labels_list: List of label lists (one per image)
            num_cols: Number of columns in the grid layout
        
        Returns:
            Result from _draw_grid (typically None after plt.show())
        """
        return self._draw_grid(bev_images, yolo_labels_list, num_cols)
    
    def _draw_grid(self, bev_images, yolo_labels_list, num_cols=2):
        """
        Internal method to render multiple BEV images with annotations in a grid.
        
        Creates a matplotlib figure with subplots arranged in a grid, showing
        multiple BEV images side-by-side for comparative analysis. This is useful
        for:
        - Dataset overview and quality assessment
        - Comparing different scenes or time points
        - Identifying patterns in object distribution
        - Spotting preprocessing issues across samples
        
        Args:
            bev_images: List of numpy arrays (BGR images)
            yolo_labels_list: List of annotation lists
            num_cols: Number of columns in grid (default: 2)
        
        Technical Details:
            1. Grid Layout:
               - Rows calculated as: ceil(num_samples / num_cols)
               - Creates num_rows × num_cols subplot grid
               - Unused subplots (if any) are hidden
            
            2. Figure Sizing:
               - Each subplot: 10×10 inches
               - Total width: 10 * num_cols inches
               - Total height: 10 * num_rows inches
               - Large size ensures readability of small objects
            
            3. Annotation Rendering:
               - Boxes drawn but labels omitted (cleaner appearance in grid)
               - Copy made of each image to avoid modifying originals
               - Color coding preserved from class definitions
            
            4. Subplot Titles:
               - Format: "Sample {index} ({count} objects)"
               - Provides quick object count per image
               - Helps identify empty vs. crowded scenes
            
            5. Edge Cases:
               - Single image: axes becomes single object, not array
               - Handles by converting to list when needed
               - Extra subplots turned off to avoid empty panels
        """
        num_samples = len(bev_images)
        
        # Calculate grid dimensions
        num_rows = (num_samples + num_cols - 1) // num_cols  # Ceiling division
        
        # Create subplot grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 10 * num_rows))
        
        # Handle single subplot case (axes is not an array)
        if num_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()  # Convert 2D grid to 1D list
        
        # Render each BEV image with annotations
        for i in range(num_samples):
            img = bev_images[i].copy()  # Copy to avoid modifying original
            h, w = img.shape[:2]
            
            # Draw bounding boxes for all objects in this image
            for label in yolo_labels_list[i]:
                class_id = int(label[0])
                
                # Denormalize coordinates
                x_center, y_center = label[1] * w, label[2] * h
                box_w, box_h = label[3] * w, label[4] * h
                
                # Convert to corner format
                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)
                
                # Draw box (no label to reduce clutter in grid view)
                cv2.rectangle(img, (x1, y1), (x2, y2), self.colors[class_id], 2)
            
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display in subplot
            axes[i].imshow(img_rgb)
            axes[i].set_title(f'Sample {i} ({len(yolo_labels_list[i])} objects)')
            axes[i].axis('off')  # Hide axis
        
        # Hide unused subplots (if grid has more cells than images)
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        # Adjust spacing and display
        plt.tight_layout()  # Reduce whitespace between subplots
        plt.show()

