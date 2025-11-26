"""
DatasetConfigGenerator: Create YOLO dataset.yaml configuration file.

This module generates the dataset.yaml file required by YOLO training, which
specifies paths to train/val/test images and defines class names. The configuration
references files directly in the preprocessed directory without duplication.

Key Operations:
- Generate dataset.yaml with absolute paths
- Define class names and IDs
- Save configuration for YOLO training
"""

from pathlib import Path
import yaml
from Globals import PREPROCESSED_ROOT, DATA_ROOT


class DatasetConfigGenerator:
    """
    Generator for YOLO dataset configuration files.
    
    Creates a dataset.yaml file that points to the preprocessed images and labels
    in their original locations, avoiding file duplication.
    
    Attributes:
        class_names: List of detection class names
        num_classes: Number of detection classes
    """
    
    def __init__(self):
        """
        Initialize the dataset config generator with class definitions.
        
        Technical Details:
            - Class IDs map to indices: 0=Car, 1=Truck/Bus, 2=Pedestrian, 3=Cyclist
            - Order must match YOLOAnnotationConverter class_mapping
        """
        self.class_names = ['car', 'truck_bus', 'pedestrian', 'cyclist']
        self.num_classes = len(self.class_names)
    
    def generate(self, splits, output_path):
        """
        Generate YOLO dataset.yaml configuration file.
        
        Creates a YAML file that references the preprocessed images and labels
        directly, with separate lists for train/val/test splits.
        
        Args:
            splits: Dictionary from DataSplitter with train/val/test file paths
            output_path: Path where dataset.yaml should be saved
        
        Returns:
            Path to the generated dataset.yaml file
        
        Technical Details:
            
            **YAML Structure:**
            ```yaml
            path: /absolute/path/to/preprocessed
            train: images/
            val: images/
            test: images/
            
            names:
              0: car
              1: truck_bus
              2: pedestrian
              3: cyclist
            
            nc: 4
            ```
            
            **Path Handling:**
            - Uses absolute paths for robustness
            - YOLO will look for labels/ in same directory as images/
            - Train/val/test specify subdirectories relative to path
            
            **File Format:**
            - Standard YAML format
            - Compatible with Ultralytics YOLO
            - Supports both v8 and v12 versions
        """
        print(f"\nGenerating dataset.yaml...")
        
        # ============================================================
        # STEP 1: Build dataset configuration dictionary
        # ============================================================
        
        # Use absolute path to preprocessed directory
        preprocessed_path = Path(PREPROCESSED_ROOT).resolve()
        
        # YOLO expects images and labels to be in parallel directories
        # DataSplitter creates train/images, val/images, test/images subdirectories
        config = {
            'path': str(preprocessed_path),
            'train': 'train/images',  # YOLO will look in path/train/images
            'val': 'val/images',      # and path/train/labels for labels
            'test': 'test/images',
            
            # Class definitions
            'names': {i: name for i, name in enumerate(self.class_names)},
            'nc': self.num_classes
        }
        
        # ============================================================
        # STEP 2: Save configuration to YAML file
        # ============================================================
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Dataset configuration saved to: {output_path}")
        
        # ============================================================
        # STEP 3: Create split manifest files for reference
        # ============================================================
        # Save train/val/test file lists for future reference
        manifest_dir = output_path.parent / 'split_manifests'
        manifest_dir.mkdir(exist_ok=True)
        
        for split_name, split_data in splits.items():
            manifest_path = manifest_dir / f'{split_name}_files.txt'
            with open(manifest_path, 'w') as f:
                # Get images from the split directory
                images_dir = split_data['images_dir']
                for img_path in sorted(images_dir.glob('*.png')):
                    # Save relative path from preprocessed root
                    # Resolve to absolute path first to match preprocessed_path
                    img_path_abs = img_path.resolve()
                    rel_path = img_path_abs.relative_to(preprocessed_path)
                    f.write(f"{rel_path}\n")
            
            print(f"  Saved {split_name} manifest: {manifest_path}")
        
        return output_path

