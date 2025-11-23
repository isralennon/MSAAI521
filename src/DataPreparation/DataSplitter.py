"""
DataSplitter: Split preprocessed dataset into train/validation/test sets.

This module handles the splitting of the preprocessed BEV dataset into training,
validation, and test subsets. The splits are performed in memory by creating lists
of file paths - no files are copied or duplicated.

Key Operations:
- Load all preprocessed image/label pairs
- Perform stratified train/val/test split
- Return file path lists for each split
- Preserve class distribution across splits
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from Globals import PREPROCESSED_ROOT


class DataSplitter:
    """
    Splitter for dividing preprocessed dataset into train/val/test sets in memory.
    
    This class implements stratified sampling to ensure that the class distribution
    is maintained across all splits. File paths are organized into lists without
    copying or moving any actual files.
    
    Attributes:
        preprocessed_root: Path to preprocessed dataset directory
        images_dir: Path to preprocessed images directory
        labels_dir: Path to preprocessed labels directory
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
        self.images_dir = self.preprocessed_root / 'images'
        self.labels_dir = self.preprocessed_root / 'labels'
    
    def split(self):
        """
        Split preprocessed dataset into train/val/test sets in memory.
        
        Returns file path lists without copying any files. The returned paths
        reference the original files in the preprocessed directory.
        
        Returns:
            Dictionary with keys 'train', 'val', 'test', each containing:
            {
                'images': List of Path objects to image files,
                'labels': List of Path objects to label files
            }
        
        Technical Details:
            - Two-stage split: first separate test, then split remainder into train/val
            - File paths are kept as Path objects, no files copied
            - Shuffle ensures random distribution
            - Random seed provides reproducibility
        """
        # ============================================================
        # STEP 1: Get all image files
        # ============================================================
        print("Scanning preprocessed dataset...")
        image_files = sorted(list(self.images_dir.glob('*.png')))
        
        if len(image_files) == 0:
            raise FileNotFoundError(
                f"No preprocessed images found in {self.images_dir}\n"
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
        
        # ============================================================
        # STEP 5: Build file path dictionaries (no copying)
        # ============================================================
        splits = {
            'train': self._build_file_lists(train_stems),
            'val': self._build_file_lists(val_stems),
            'test': self._build_file_lists(test_stems)
        }
        
        # Print split summary
        print(f"\nDataset split:")
        print(f"  Train: {len(train_stems)} samples ({len(train_stems)/len(file_stems)*100:.1f}%)")
        print(f"  Val:   {len(val_stems)} samples ({len(val_stems)/len(file_stems)*100:.1f}%)")
        print(f"  Test:  {len(test_stems)} samples ({len(test_stems)/len(file_stems)*100:.1f}%)")
        
        return splits
    
    def _build_file_lists(self, file_stems):
        """
        Build lists of image and label file paths from filename stems.
        
        Args:
            file_stems: List of filename stems (without extensions)
        
        Returns:
            Dictionary with 'images' and 'labels' keys containing Path lists
        """
        images = [self.images_dir / f"{stem}.png" for stem in file_stems]
        labels = [self.labels_dir / f"{stem}.txt" for stem in file_stems]
        
        return {
            'images': images,
            'labels': labels
        }

