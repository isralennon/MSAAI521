# MSAAI521
USD MS AAI - 521 - Fall 2025 - Final Project, Team 2

## Setup

You need to get the data from https://www.nuscenes.org/nuscenes#download. Get: 

Full dataset (v1.0) -> Mini (3.88GB).

Create a build/data/raw folder at the root of the project and place the unzipped files in there. It should be:

```
MSAAI521/
├── build/
│   ├── data/
│       └── raw/
│           └── v1.0-mini/
│               ├── samples/
│               ├── sweeps/
│               └── v1.0-mini/
│                   └── ...
├── src/
│   └── ...
├── demo_notebook.ipynb
└── README.md
```

Don't forget to run:

```
pip install -r requirements.txt
```

## Usage

Run the full pipeline from the `src/` directory:

```bash
cd src
python main.py
```

The pipeline executes the following stages sequentially:
1. **Data Validation** — Verifies dataset integrity
2. **Preprocessing** — Converts point clouds to BEV images and annotations to YOLO format
3. **Data Splitting** — Creates train/val/test splits (70/15/15)
4. **Training Stage 1** — Warm-up with frozen backbone (50 epochs)
5. **Training Stage 2** — Fine-tuning all layers (150 epochs)
6. **Evaluation** — Computes mAP, precision, recall metrics
7. **Visualization** — Generates prediction visualizations

---

## Implementation Details

**Software Environment:**
- Python 3.10
- PyTorch 2.0.0
- Ultralytics YOLOv12 8.0.0
- nuScenes DevKit 1.1.9
- CUDA 11.8 with cuDNN 8.6

**Repository Structure:**
```
MSAAI521/
├── src/
│   ├── DataDownload/       # Dataset acquisition and validation
│   ├── Preprocessing/      # BEV conversion and annotation processing
│   ├── DataPreparation/    # Train/val/test splitting
│   ├── Training/           # Model training orchestration
│   └── Evaluation/         # Metrics computation and visualization
├── build/
│   ├── data/               # Preprocessed dataset
│   ├── models/             # Trained model checkpoints
│   └── results/            # Evaluation outputs
└── requirements.txt
```

## Preprocessing Parameters

**BEV Rasterization:**
- X-range: [-50m, 50m]
- Y-range: [-50m, 50m]
- Z-range: [-3m, 5m]
- Resolution: 0.09765625m per pixel (1024×1024) or 0.078125m per pixel (1280×1280)
- Output size: 1024 × 1024 × 3 or 1280 × 1280 × 3

**Coordinate Transformations:**
- Sensor frame: LIDAR_TOP
- Transformation chain: Global → Ego → Sensor
- Quaternion-based rotations
- Translation vectors from calibration

## Training Hyperparameters

**Stage 1 (Warm-up):**
- Epochs: 50
- Learning rate: 0.01
- Frozen layers: 10 (backbone)
- Batch size: 16

**Stage 2 (Fine-tuning):**
- Epochs: 150
- Learning rate: 0.001
- Frozen layers: 0 (all trainable)
- Early stopping patience: 50

**Data Augmentation:**
- Rotation: ±15°
- Translation: ±10%
- Scale: ±50%
- Horizontal flip: 50%
- Mosaic: 100%
- MixUp: 10%

## Computational Requirements

**Training:**
- GPU: NVIDIA RTX 3090 (24 GB)
- Training time: 2-4 hours (v1.0-mini)
- Memory usage: ~12 GB

**Inference:**
- GPU: NVIDIA RTX 3090
- Throughput: 40-60 FPS
- Latency: 16-25 ms per frame

**Storage:**
- Raw dataset: 4 GB (v1.0-mini)
- Preprocessed BEV: ~1.2 GB
- Model checkpoints: ~20 MB per checkpoint

## Key Modules

- `BEVRasterizer.py`: Point cloud to BEV conversion
- `YOLOAnnotationConverter.py`: 3D to 2D annotation transformation
- `TrainingOrchestrator.py`: Two-stage training pipeline
- `ModelEvaluator.py`: Metrics computation and visualization

