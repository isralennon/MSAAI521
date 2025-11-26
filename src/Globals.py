import os

BUILD_ROOT = "build"

DATA_ROOT = f"{BUILD_ROOT}/data"

# Auto-detect nuScenes version (supports both v1.0-mini and v1.0-trainval)
# Priority: v1.0-trainval (full dataset) if exists, otherwise v1.0-mini
_NUSCENES_VERSIONS = ["v1.0-trainval", "v1.0-mini"]
NUSCENES_VERSION = None
NUSCENES_ROOT = None

for version in _NUSCENES_VERSIONS:
    candidate_path = f"{DATA_ROOT}/raw/{version}"
    if os.path.exists(candidate_path):
        NUSCENES_VERSION = version
        NUSCENES_ROOT = candidate_path
        break

# Fallback if neither exists (for initial setup)
if NUSCENES_VERSION is None:
    NUSCENES_VERSION = "v1.0-mini"
    NUSCENES_ROOT = f"{DATA_ROOT}/raw/{NUSCENES_VERSION}"

PREPROCESSED_ROOT = f"{DATA_ROOT}/preprocessed"
YOLO_BEV_ROOT = f"{DATA_ROOT}/yolo_bev"

MODELS_ROOT = f"{BUILD_ROOT}/models"

RUNS_ROOT = f"{BUILD_ROOT}/runs"
RESULTS_ROOT = f"{BUILD_ROOT}/results"

VISUALIZATIONS_ROOT = f"{BUILD_ROOT}/visualizations"

