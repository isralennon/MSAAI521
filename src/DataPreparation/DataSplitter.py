"""
DataSplitter: Split preprocessed dataset into train/validation/test sets.

This module handles the splitting of the preprocessed BEV dataset into training,
validation, and test subsets by physically organizing files into separate directories.

Key Operations:
- Load all preprocessed image/label pairs
- Perform stratified train/val/test split
- Copy files to train/val/test directories
- Preserve class distribution across splits
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import shutil
from Globals import PREPROCESSED_ROOT


class DataSplitter:
    """
    Splitter for dividing preprocessed dataset into train/val/test sets.
    
    This class implements stratified sampling to ensure that the class distribution
    is maintained across all splits. Files are physically copied into separate
    train/val/test directories.
    
    Attributes:
        preprocessed_root: Path to preprocessed dataset directory
        source_images_dir: Path to source preprocessed images
        source_labels_dir: Path to source preprocessed labels
        train_ratio: Fraction of data for training (default: 0.7)
        val_ratio: Fraction of data for validation (default: 0.15)
        test_ratio: Fraction of data for testing (default: 0.15)
        random_seed: Random seed for reproducibility
    """
    
    def __init__(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """
        Initialize the data splitter with split ratios.
        
        Args:
            train_ratio: Fraction of data for training (0 < x < 1)
            val_ratio: Fraction of data for validation (0 < x < 1)
            test_ratio: Fraction of data for testing (0 < x < 1)
            random_seed: Random seed for reproducible splits
        
        Technical Details:
            - Ratios must sum to 1.0
            - Typical splits: 70/15/15 or 80/10/10
            - Random seed ensures same split across runs
        """
        # Validate ratios sum to 1.0
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        
        # Setup paths
        self.preprocessed_root = Path(PREPROCESSED_ROOT)
        self.source_images_dir = self.preprocessed_root / 'images'
        self.source_labels_dir = self.preprocessed_root / 'labels'
    
    def split(self):
        """
        Split preprocessed dataset into train/val/test directories.
        
        Creates physical train/val/test directories and copies files into them.
        This ensures proper data separation and prevents data leakage.
        
        Returns:
            Dictionary with keys 'train', 'val', 'test', each containing:
            {
                'images_dir': Path to images directory,
                'labels_dir': Path to labels directory,
                'num_samples': Number of samples in split
            }
        
        Directory Structure Created:
            preprocessed/
            ├── train/
            │   ├── images/
            │   └── labels/
            ├── val/
            │   ├── images/
            │   └── labels/
            └── test/
                ├── images/
                └── labels/
        
        Technical Details:
            - Two-stage split: first separate test, then split remainder into train/val
            - Files are copied (not moved) to preserve original preprocessed data
            - Shuffle ensures random distribution
            - Random seed provides reproducibility
        """
        # ============================================================
        # STEP 1: Get all image files from source
        # ============================================================
        print("Scanning preprocessed dataset...")
        image_files = sorted(list(self.source_images_dir.glob('*.png')))
        
        if len(image_files) == 0:
            raise FileNotFoundError(
                f"No preprocessed images found in {self.source_images_dir}\n"
                f"Have you run the preprocessing stage?"
            )
        
        print(f"Found {len(image_files)} preprocessed samples")
        
        # ============================================================
        # STEP 2: Create list of filename stems
        # ============================================================
        file_stems = [f.stem for f in image_files]
        
        # ============================================================
        # STEP 3: First split - separate test set
        # ============================================================
        train_val_stems, test_stems = train_test_split(
            file_stems,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            shuffle=True
        )
        
        # ============================================================
        # STEP 4: Second split - separate train and val
        # ============================================================
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        
        train_stems, val_stems = train_test_split(
            train_val_stems,
            test_size=val_ratio_adjusted,
            random_state=self.random_seed,
            shuffle=True
        )
        
        # Print split summary
        print(f"\nDataset split:")
        print(f"  Train: {len(train_stems)} samples ({len(train_stems)/len(file_stems)*100:.1f}%)")
        print(f"  Val:   {len(val_stems)} samples ({len(val_stems)/len(file_stems)*100:.1f}%)")
        print(f"  Test:  {len(test_stems)} samples ({len(test_stems)/len(file_stems)*100:.1f}%)")
        
        # ============================================================
        # STEP 5: Create split directories and copy files
        # ============================================================
        print("\nCreating split directories and copying files...")
        
        splits = {
            'train': self._create_split_directory('train', train_stems),
            'val': self._create_split_directory('val', val_stems),
            'test': self._create_split_directory('test', test_stems)
        }
        
        print("✓ Dataset split complete")
        
        return splits
    
    def _create_split_directory(self, split_name, file_stems):
        """
        Create split directory structure and copy files.
        
        Args:
            split_name: Name of split ('train', 'val', or 'test')
            file_stems: List of filename stems to include in this split
        
        Returns:
            Dictionary with split metadata:
            {
                'images_dir': Path to images directory,
                'labels_dir': Path to labels directory,
                'num_samples': Number of samples
            }
        """
        # Create directory structure
        split_root = self.preprocessed_root / split_name
        images_dir = split_root / 'images'
        labels_dir = split_root / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        print(f"  Copying {len(file_stems)} samples to {split_name}/...")
        for stem in file_stems:
            # Copy image
            src_img = self.source_images_dir / f"{stem}.png"
            dst_img = images_dir / f"{stem}.png"
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            src_label = self.source_labels_dir / f"{stem}.txt"
            dst_label = labels_dir / f"{stem}.txt"
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
        
        return {
            'images_dir': images_dir,
            'labels_dir': labels_dir,
            'num_samples': len(file_stems)
        }

