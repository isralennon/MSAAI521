"""
ModelInitializer: Initialize YOLO model with pretrained weights.

This module handles loading the YOLOv12 model with pretrained COCO weights for
transfer learning. It downloads weights if needed and prepares the model for training.

Key Operations:
- Load YOLOv12 with pretrained weights
- Validate model architecture
- Prepare for transfer learning
"""

from ultralytics import YOLO
from pathlib import Path
from Globals import MODELS_ROOT


class ModelInitializer:
    """
    Initializer for YOLO models with transfer learning support.
    
    This class handles model instantiation, pretrained weight loading, and
    configuration for fine-tuning on the BEV detection task.
    
    Attributes:
        model_size: YOLO model size variant ('n', 's', 'm', 'l')
        pretrained: Whether to use pretrained weights
        models_dir: Directory for storing model weights
    """
    
    def __init__(self, model_size='s', pretrained=True):
        """
        Initialize the model initializer.
        
        Args:
            model_size: Model variant - 'n' (nano), 's' (small), 'm' (medium), 'l' (large)
            pretrained: Load COCO pretrained weights (True) or random initialization (False)
        
        Technical Details:
            - 's' (small) provides good balance of speed and accuracy
            - Pretrained weights improve convergence and final performance
            - Models are cached in build/models/ directory
        """
        self.model_size = model_size
        self.pretrained = pretrained
        self.models_dir = Path(MODELS_ROOT)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize(self):
        """
        Initialize YOLO model with optional pretrained weights.
        
        Downloads pretrained weights if needed and creates a YOLO model instance
        ready for training on the BEV detection task.
        
        Returns:
            YOLO model instance configured for training
        
        Technical Details:
            
            **Model Variants:**
            - YOLOv12n: 1.9M params, fastest, lowest accuracy
            - YOLOv12s: 9.1M params, balanced (recommended)
            - YOLOv12m: 23.8M params, higher accuracy, slower
            - YOLOv12l: 52.6M params, highest accuracy, slowest
            
            **Transfer Learning:**
            - Pretrained weights are from COCO dataset (80 classes)
            - Detection head will be replaced with 4-class head automatically
            - Backbone and neck preserve learned features
            - Significant speedup in convergence vs random initialization
            
            **Weight Management:**
            - Weights downloaded to build/models/ on first use
            - Subsequent runs use cached weights
            - Network connection required only for first download
        """
        print(f"\nInitializing YOLOv12{self.model_size} model...")
        
        # ============================================================
        # STEP 1: Construct model name
        # ============================================================
        if self.pretrained:
            # Pretrained weights format: yolo12s.pt
            model_name = f'yolo12{self.model_size}.pt'
            print(f"  Loading with COCO pretrained weights")
        else:
            # Architecture config format: yolo12s.yaml
            model_name = f'yolo12{self.model_size}.yaml'
            print(f"  Random initialization (no pretrained weights)")
        
        # ============================================================
        # STEP 2: Initialize model
        # ============================================================
        # YOLO() automatically downloads weights if not found
        model = YOLO(model_name)
        
        # ============================================================
        # STEP 3: Print model info
        # ============================================================
        # Count total parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        print(f"  Model: YOLOv12{self.model_size}")
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"  Pretrained: {self.pretrained}")
        
        return model

