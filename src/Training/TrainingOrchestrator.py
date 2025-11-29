"""
TrainingOrchestrator: Orchestrate two-stage YOLO training pipeline.

This module manages the complete training workflow including warm-up stage with
frozen backbone and fine-tuning stage with all layers trainable. It handles
hyperparameter configuration, training execution, and checkpoint management.

Key Operations:
- Configure training hyperparameters for each stage
- Execute two-stage training (warm-up + fine-tuning)
- Manage checkpoints and logging
- Monitor training progress
"""

from pathlib import Path
from Globals import RUNS_ROOT


class TrainingOrchestrator:
    """
    Orchestrator for two-stage YOLO training pipeline.
    
    Manages the complete training workflow with separate warm-up and fine-tuning
    stages to optimize transfer learning from COCO to BEV detection task.
    
    Attributes:
        model: YOLO model instance
        dataset_yaml: Path to dataset configuration file
        runs_dir: Directory for training outputs
    """
    
    def __init__(self, model, dataset_yaml):
        """
        Initialize the training orchestrator.
        
        Args:
            model: YOLO model instance from ModelInitializer
            dataset_yaml: Path to dataset.yaml configuration file
        
        Technical Details:
            - Training outputs saved to build/runs/
            - Separate directories for each training stage
            - Tensorboard logs automatically generated
        """
        self.model = model
        self.dataset_yaml = str(dataset_yaml)
        self.runs_dir = Path(RUNS_ROOT)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
    
    def train_stage1_warmup(self, epochs=50, batch_size=4, img_size=1024):
        """
        Stage 1: Warm-up training with frozen backbone.
        
        Trains only the detection head while keeping the backbone frozen. This
        allows the head to adapt to the new domain (BEV detection) without
        disrupting pretrained feature extraction.
        
        Args:
            epochs: Number of training epochs (default: 50)
            batch_size: Batch size (default: 4, optimized for 1024px images)
            img_size: Input image size (default: 1024 to match BEV resolution)
        
        Returns:
            Training results object from YOLO
        
        Technical Details:
            
            **Freezing Strategy:**
            - freeze=10: Freezes first 10 layers (backbone)
            - Detection head and neck remain trainable
            - Prevents catastrophic forgetting of low-level features
            
            **Learning Rate:**
            - lr0=0.01: Higher initial LR acceptable since only head trains
            - Cosine annealing reduces LR over epochs
            - Warm-up helps stabilize early training
            
            **Data Augmentation:**
            - Mosaic: Combines 4 images for better small object detection
            - MixUp: Blends images to improve generalization
            - Geometric: Rotation, translation, scale for robustness
        """
        print("\n" + "="*80)
        print("STAGE 1: WARM-UP TRAINING (Frozen Backbone)")
        print("="*80)
        
        # ============================================================
        # Configure Stage 1 hyperparameters
        # ============================================================
        config = {
            # Dataset
            'data': self.dataset_yaml,
            'imgsz': img_size,
            'batch': batch_size,
            
            # Training duration
            'epochs': epochs,
            
            # Optimizer
            'optimizer': 'AdamW',
            'lr0': 0.01,          # Initial learning rate
            'lrf': 0.01,          # Final LR (fraction of lr0)
            'momentum': 0.937,
            'weight_decay': 0.0005,
            
            # Learning rate schedule
            'cos_lr': True,       # Cosine annealing
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Freezing
            'freeze': 10,         # Freeze first 10 layers (backbone)
            
            # Data augmentation
            'degrees': 15.0,      # Rotation
            'translate': 0.1,     # Translation
            'scale': 0.5,         # Scale
            'fliplr': 0.5,        # Horizontal flip
            'mosaic': 0.5,        # Mosaic augmentation
            'mixup': 0.0,         # MixUp augmentation
            
            # Hardware
            'device': 0,          # GPU 0 (use 'cpu' for CPU training)
            'workers': 2,         # DataLoader workers
            'cache': False,
            'amp': True,
            # Logging and saving
            'project': str(self.runs_dir / 'detect'),
            'name': 'stage1_warmup',
            'exist_ok': True,
            'save': True,
            'save_period': 10,    # Save checkpoint every 10 epochs
            'plots': True,
            'verbose': True,
            
            # Other
            'seed': 42,
            'deterministic': True,
            'val': True,          # Run validation
        }
        
        # ============================================================
        # Execute training
        # ============================================================
        print(f"\nStarting Stage 1 training...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {img_size}")
        print(f"  Frozen layers: 10 (backbone)")
        
        results = self.model.train(**config)
        
        print(f"\n✓ Stage 1 complete")
        print(f"  Results saved to: {results.save_dir}")
        print(f"  Best weights: {results.save_dir}/weights/best.pt")
        
        return results
    
    def train_stage2_finetune(self, stage1_weights_path, epochs=150, batch_size=4, img_size=1024):
        """
        Stage 2: Fine-tuning with all layers trainable.
        
        Unfreezes all layers and trains end-to-end with reduced learning rate.
        This allows the entire network to adapt to BEV-specific features while
        maintaining learned representations.
        
        Args:
            stage1_weights_path: Path to best weights from Stage 1
            epochs: Number of training epochs (default: 150)
            batch_size: Batch size (default: 4, optimized for 1024px images)
            img_size: Input image size (default: 1024 to match BEV resolution)
        
        Returns:
            Training results object from YOLO
        
        Technical Details:
            
            **Unfreezing Strategy:**
            - freeze=0: All layers trainable
            - Lower LR prevents catastrophic forgetting
            - Allows fine-grained adaptation to BEV domain
            
            **Learning Rate:**
            - lr0=0.001: 10x lower than Stage 1
            - Prevents disrupting learned features
            - Enables careful fine-tuning
            
            **Early Stopping:**
            - patience=50: Stop if no improvement for 50 epochs
            - Prevents overfitting
            - Saves computational resources
        """
        print("\n" + "="*80)
        print("STAGE 2: FINE-TUNING (All Layers Trainable)")
        print("="*80)
        
        # ============================================================
        # Load best model from Stage 1
        # ============================================================
        from ultralytics import YOLO
        model = YOLO(stage1_weights_path)
        print(f"Loaded Stage 1 weights from: {stage1_weights_path}")
        
        # ============================================================
        # Configure Stage 2 hyperparameters
        # ============================================================
        config = {
            # Dataset
            'data': self.dataset_yaml,
            'imgsz': img_size,
            'batch': batch_size,
            
            # Training duration
            'epochs': epochs,
            
            # Optimizer
            'optimizer': 'AdamW',
            'lr0': 0.001,         # Lower LR for fine-tuning
            'lrf': 0.001,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            
            # Learning rate schedule
            'cos_lr': True,
            'warmup_epochs': 0,   # No warmup needed
            
            # Freezing
            'freeze': 0,          # Unfreeze all layers
            
            # Data augmentation (same as Stage 1)
            'degrees': 15.0,
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            'mosaic': 0.5,
            'mixup': 0.0,
            
            # Hardware
            'device': 0,
            'workers': 2,         # DataLoader workers
            'cache': False,
            'amp': True,
            
            # Logging and saving
            'project': str(self.runs_dir / 'detect'),
            'name': 'stage2_finetune',
            'exist_ok': True,
            'save': True,
            'save_period': 10,
            'plots': True,
            'verbose': True,
            
            # Early stopping
            'patience': 50,       # Stop if no improvement for 50 epochs
            
            # Other
            'seed': 42,
            'deterministic': True,
            'val': True,
        }
        
        # ============================================================
        # Execute training
        # ============================================================
        print(f"\nStarting Stage 2 training...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {img_size}")
        print(f"  All layers trainable")
        
        results = model.train(**config)
        
        print(f"\n✓ Stage 2 complete")
        print(f"  Results saved to: {results.save_dir}")
        print(f"  Best weights: {results.save_dir}/weights/best.pt")
        
        return results

    def train_stage1(self, epochs=50, batch_size=4):
        """
        Train stage 1 of the pipeline.

        Args:
            epochs: Number of training epochs (default: 50)
            batch_size: Batch size (default: 4, optimized for 1024px images)

        Returns:
            Training results object from YOLO
        """
        return self.train_stage1_warmup(
            epochs=epochs,
            batch_size=batch_size
        )

    def train_stage2(self, stage1_weights_path, epochs=150, batch_size=4):
        """
        Train stage 2 of the pipeline.

        Args:
            stage1_weights_path: Path to best weights from Stage 1
            epochs: Number of training epochs (default: 150)
            batch_size: Batch size (default: 4, optimized for 1024px images)

        Returns:
            Training results object from YOLO
        """
        return self.train_stage2_finetune(
            stage1_weights_path=stage1_weights_path,
            epochs=epochs,
            batch_size=batch_size
        )

