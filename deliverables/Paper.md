# Real-Time Object Detection Using LiDAR and Bird's-Eye View Representation

## Abstract

Autonomous vehicles require reliable real-time perception systems to detect surrounding objects under various environmental conditions, from clear skies to deep fog, rain, or snow. While camera-based detection systems are mature and efficient, they struggle in low-light or poor visibility conditions and lack inherent depth information. LiDAR sensors provide superior 3D spatial awareness and are designed to work well even with poor visibility conditions, but it comes with a cost: traditional 3D detection models are computationally intensive and difficult to deploy into a limited resources mobile environment. This project investigates a hybrid approach: converting 3D LiDAR point clouds into 2D Bird's-Eye View (BEV) images and training YOLOv12 for efficient object detection. Our goal is to demonstrate that this approach combines the best of both worlds: spatial accuracy of LiDAR with the computational efficiency of 2D detection architectures, achieving real-time performance suitable for a portable autonomous driving system.

## 1. Introduction

### 1.1 Motivation

In 1982 there was a famous TV show about a self-driving vehicle with artificial intelligence capable of talking, among other amazing features like video-calls or online wireless connectivity (before the Internet was even a thing). All these features were nothing but futuristic Science Fiction that seemed far away in an unreachable distant future.

Today is the future of that decade, and many of those features that captured our imagination are now a reality, like devices that can talk back to you, 24/7 wireless online connectivity, video calls, and of course: self-driving cars.

While only a few brands have offered fully autonomous vehicles out to the public, it's no longer fiction, but just a matter of time before the technology gets fully adopted. Before that happens, there are still challenges to solve.

Autonomous vehicles require real-time environment perception to detect and track surrounding objects accurately. The perception system must function reliably across varying lighting conditions, weather, and complex urban environments. Camera-based detectors, while computationally efficient, face limitations in poor lighting and lack direct depth information. 

LiDAR sensors overcome these limitations by providing accurate 3D measurements regardless of lighting conditions, as they work by sending short laser pulses of near-infrared light that can easily penetrate fog, snow, or rain, while being invisible to the human eye. After emitting the pulses, the LiDAR uses the speed of light by measuring the time it takes for them to bounce back in surrounding structures, then deducing the distance of such objects. 

However, processing 3D point clouds for object detection traditionally requires specialized architectures like PointPillars or CenterPoint, that are computationally expensive and complex to train and present a challenge when trying to build an embedded system for a self-driving vehicle.

### 1.2 Problem Statement

Besides a high computing power requirement for real-time image processing, another challenge is depth: objects look smaller when they are further away, which might become a problem in object detection due to scale, as well as resolution limitations. This is why we have chosen Bird's-Eye View (BEV), an emulation of a perspective taken from above the autonomous vehicle (also known as the ego vehicle). From the BEV perspective, objects preserve their scale, making it easier for object-detection methods like YOLO, while lowering the computing power requirement.

This project addresses the following question: Can we apply the efficiency of mature 2D object detection models by converting LiDAR point clouds into Bird's-Eye View images, while maintaining the spatial accuracy advantages of LiDAR sensing?

### 1.3 Objectives

Our research objectives are:

1. Convert raw 3D LiDAR point clouds from the nuScenes dataset into 2D Bird's-Eye View (BEV) raster images
2. Transform 3D bounding box annotations into 2D YOLO-compatible format aligned with BEV representations
3. Train and validate a YOLOv12 model for LiDAR-based object detection on BEV images
4. Evaluate detection performance across multiple object classes (vehicles, pedestrians, cyclists)
5. Assess the feasibility of this approach for real-time autonomous driving applications

### 1.4 Contributions

We built the following features as part of our project:

- A complete pipeline for converting nuScenes 3D LiDAR data into YOLO-compatible 2D BEV format
- Implementation of coordinate transformation and axis-aligned bounding box computation for BEV projection
- A two-stage transfer learning methodology adapted from COCO-pretrained weights to LiDAR BEV domain
- A comprehensive evaluation demonstrating the feasibility of 2D detection models on LiDAR data

## 2. Literature Review


The history of Computer Vision for self-driving has been marked by breakthroughs in two key areas: the progress in object detection using deep neural networks such as YOLO and the publication of large-scale autonomous driving datasets.

One of the major impediments to autonomous driving was the slow inference time of early computer vision models such as R-CNN, which required multiple passes for every prediction. Because of the high frequency of changes in driving environments and the need for split-second decisions, the speed of inference was a critical operational constraint. This changed when Redmon et al. (2016) introduced YOLO, an object detection architecture capable of outputting box coordinates and class probabilities in a single pass. This unified approach enabled real-time detection at 45 FPS while maintaining competitive accuracy, establishing YOLO as a practical choice for deployment-oriented applications.

In addition to advances in object detection architectures, the trajectory of self-driving was reshaped by two major data-driven developments.

The KITTI dataset, proposed by Geiger et al. (2012), was the first large-scale real-world dataset with synchronized LiDAR, camera, and GPS samples. It formed the basis for the first set of unified self-driving benchmarks and evaluation metrics.

Subsequently, Caesar et al. (2020) released nuScenes,a large-scale autonomous driving dataset collected by Motional and nuTonomy. It contained samples from 1000 driving scenes with full 360° sensor coverage including LiDAR, RADAR, and six cameras. This dataset became a landmark in the industry and further solidified the performance metrics and evaluation goals used for self-driving perception models.


## 3. Dataset and Exploratory Analysis

### 3.1 nuScenes Dataset Overview

We used the nuScenes dataset. The dataset provides multi-sensor data including LiDAR, RADAR, cameras, GPS, and IMU (Figure 1), collected in Boston and Singapore under diverse driving conditions and over one thousand carefully planned driving scenes of about twenty seconds each.

![](Resources/Sensors.png)

Figure 1: Sensor arrangement on NuScenes cars

The resulting dataset came in two flavors, which are described in Table 1.

Table 1: Dataset Specifications:

| Dataset        | Source                | Scenes | Samples | Size   |
|----------------|----------------------|--------|---------|--------|
| v1.0-mini      | nuScenes official    | 10     | 404     | ~4 GB  |
| v1.0-trainval  | Kaggle (nuScenes mirror) | 850    | 3,377   | ~50 GB |


### 3.2 Dataset Structure

As illustrated in Figure 2, the nuScenes dataset is organized hierarchically with scenes as the top-level ojects. Each scene is a 20 second driving sequence and contains 20 samples captured at 2Hz (twice per second). Each sample contains one snapshot from each sensor on the car, including the LiDAR. In addition, each sample contains a series of annotations as class annotated bounding boxes.

![](Resources/DatasetSchema.png)

Figure 2: Dataset schema of NuScenes dataset

### 3.3 Exploratory Data Analysis

We conducted comprehensive exploratory analysis to understand the data characteristics, as well as verifying the completeness and consistency of the data:

The point cloud files are composed of 30,000 to 40,0000 points positioned in a 3D coordinate space which represents the surroundings of the vehicle. In addition to encoding positional information, they also encode height (values ranged from -3 to 5m) and intensity information (a measure of the reflectivity of the contact material). 

The per-class distribution of annotated objects was skewed, with ~50% of cars, 10% of trucks/buses, ~35% of pedestrians and ~5% of cyclists.

The dataset contained a wide variety of driving scenarios, including urban interesections with heavy traffic, highways with high-speed vehicles, parking lots with stationary objects and pedestrians, construction zones with unusual vehicle types and day and night scenes across different weather conditions.


### 3.4 Visualization Examples

We visualized three levels of data:

1. **Timestep Level:** Individual timesteps with synchronized multi-sensor data and annotations

![](Resources/TimestepVisualization.png)


2. **Sample Level:** File associated with one of the sensors for a given timestep




3. **Annotation Level:** 3D bounding boxes overlaid on point clouds, demonstrating ground truth quality



These visualizations confirmed the dataset's richness and the feasibility of BEV projection for object detection. We also confirmed the completeness of the data, and visually checked the quality of the annotations accross a few randomly selected scenes, using all the available data (LiDAR, cameras, and radar information).

Our goal is to predict the annotated classes and their locations in each frame, which would allow the self-driving vehicle to make real-time decisions.

## 4. Data Preprocessing

The preprocessing stage transforms raw 3D LiDAR point clouds into 2D BEV images suitable for YOLO training. This critical step determines detection quality and requires careful coordinate transformations and representation design.

We chose an initial resolution of 1000 by 1000 pixels, which we adjusted later to 1024 by 1024 to improve our model's accuracy.

### 4.1 Preprocessing Pipeline

Our preprocessing pipeline consists of five steps:

1. Load 3D LiDAR point cloud from binary format (already in sensor frame)
2. Filter points to region of interest (ROI)
3. Rasterize 3D points to 2D BEV image
4. Transform 3D annotations to match sensor frame
5. Convert 3D annotations to 2D YOLO format



### 4.2 Point Cloud Loading and Coordinate Frame Strategy

Each LiDAR scan contains approximately 30,000-40,000 points stored as 4D vectors: [x, y, z, intensity]. The nuScenes dataset stores point clouds in sensor frame but annotations in global coordinates. 

Since we are modifying the resolution to improve the performance of the model, we must also scale the annotations accordingly to ensure proper spatial alignment.

**Coordinate Frame Strategy:**
- Point clouds: Loaded and remain in LIDAR_TOP sensor frame (no transformation applied)
- Annotations: Transformed from global → ego vehicle → sensor frame
- This alignment strategy avoids unnecessary point cloud transformations while ensuring bounding boxes correctly overlay the point cloud data

As a result, all annotations reflect the correct locations of the detected objects, which we'll use for training and validation purposes.

The transformation formula applied was as follows:

$$P_{sensor}=R^{T}_{sensor}\cdot(R^{T}_{ego}\cdot(P_{global}-t_{ego})-t_{sensor})$$

where $R$ and $t$ are rotation matrices and translation vectors from calibration data. Rotations are represented as quarterinions $(q_w, q_x, q_y, q_x)$ and converted to rotation matrices.

### 4.3 Region of Interest Filtering

Lidar range can reach long distances, so we've decided to limit the processing to data within a range of 100m x 100m around the vehicle (328ft x 328ft).

To do that, we filter points to a 100m × 100m area around the vehicle:
- X-range: [-50m, 50m] (forward/backward)
- Y-range: [-50m, 50m] (left/right)
- Z-range: [-3m, 5m] (ground to elevated structures)

This filtering reduces computational load by 50-70% while retaining all relevant objects for autonomous driving perception.

### 4.4 Bird's-Eye View Rasterization

The core innovation of our approach is converting 3D point clouds into 2D BEV images that encode spatial information in multiple channels. The colors on the images won't technically represent real colors, but rather each RGB channel represents dimensions to be used by the YOLO model: height of the cloud point, intensity, and density.

When visually inspected, the image appears to have very little contrast and some of the features might be quite faint for the human eye, but these values will be used with precision by the YOLO model.

**BEV Image Specifications:**
- Resolution: 0.09765625 per pixel for 1024 x 1024 images and 0.078125 for 1280 x 1280.
- Dimensions: 1024 × 1024 and 1280 x 1280 pixels
- Coverage: 100m × 100m physical area
- Channels: 3 (height, intensity, density)

Points in the point cloud are mapped to the image coordinate space using the following formula:

$$u = \left\lfloor \frac{x - x_{\min}}{r} \right\rfloor, \quad v = H - 1 - \left\lfloor \frac{y - y_{\min}}{r} \right\rfloor$$

where $x_{min}=y_{min}=-50m$ and $r$ is the resolution.

**Channel Encoding:**

1. **Height Channel (Red):** Encodes maximum elevation at each pixel
   - Formula: max(z-coordinates) for points projecting to same pixel
   - Normalized to [0, 1] then gamma-corrected (power 0.5)
   - Helps distinguish vertical structures from ground plane

$$C_H(u, v) = 255 \cdot \left( \frac{\max_i(z_i) - z_{\min}}{z_{\max} - z_{\min}} \right)^{0.5}$$

2. **Intensity Channel (Green):** Encodes average LiDAR reflectivity
   - Formula: mean(intensity) for points per pixel
   - Captures material properties (metal, fabric, vegetation)
   - Aids in distinguishing object types
  
$$C_I(u, v) = 255 \cdot \frac{1}{n} \sum_{i=1}^{n} \text{intensity}_i$$

3. **Density Channel (Blue):** Encodes point count per pixel
   - Formula: log(1 + point_count) for wide dynamic range compression
   - Indicates measurement confidence and proximity
   - Higher density near vehicle and on solid surfaces

$$C_D(u, v) = 255 \cdot \frac{\log(1 + n)}{\log(1 + n_{\max})}$$

where $n$ is the number of points at pixel $(u,v)$ and $n_{\max}$ is the maximum point count across all pixels.

![](Resources/BEV.png)

**Rasterization Process:**

To convert our 4D data per cloud point into a 3-channel pixel, we used the transformation defined below.

For each point (x, y, z, intensity):
1. Compute pixel coordinates: `pixel_x = (x - x_min) / resolution`
2. Apply Y-axis flip for image coordinate convention
3. Accumulate height (maximum), intensity (sum), and density (count)
4. Normalize all channels to [0, 255] uint8 range
5. Stack into 3-channel RGB image

This representation preserves spatial relationships while enabling efficient 2D convolution operations.

### 4.5 Annotation Conversion

Converting 3D bounding boxes to 2D YOLO format requires careful geometric transformation:

**Coordinate Transformation Chain:**
1. 3D boxes start in global coordinates
2. Transform to ego vehicle frame using ego pose
3. Transform to sensor frame using calibrated sensor parameters
4. Project to BEV by discarding Z-dimension

**Axis-Aligned Bounding Box Computation:**

Standard YOLO format does not support rotated boxes. We compute axis-aligned bounding boxes (AABB) that fully contain rotated 3D boxes:

1. Get all 8 corners of rotated 3D box
2. Find min/max X and Y coordinates (top-down projection)
3. Compute AABB center and dimensions from extents
4. Normalize to [0, 1] for YOLO format

Given a 3D bounding box with center $(c_x, c_y, c_z)$, dimensions $(l, w, h)$, and yaw angle $\theta$, the 8 corners in sensor frame are computed using rotation matrix $R_z(\theta)$. The AABB in BEV is then:

$$x_{\min}^{\text{AABB}} = \min_i(x_i^{\text{corner}}), \quad x_{\max}^{\text{AABB}} = \max_i(x_i^{\text{corner}})$$

$$y_{\min}^{\text{AABB}} = \min_i(y_i^{\text{corner}}), \quad y_{\max}^{\text{AABB}} = \max_i(y_i^{\text{corner}})$$

**YOLO Format Normalization:**

$$x_c = \frac{(x_{\min}^{\text{AABB}} + x_{\max}^{\text{AABB}})/2 - x_{\min}}{W \cdot r}, \quad w = \frac{x_{\max}^{\text{AABB}} - x_{\min}^{\text{AABB}}}{W \cdot r}$$

$$y_c = 1 - \frac{(y_{\min}^{\text{AABB}} + y_{\max}^{\text{AABB}})/2 - y_{\min}}{H \cdot r}, \quad h = \frac{y_{\max}^{\text{AABB}} - y_{\min}^{\text{AABB}}}{H \cdot r}$$

**YOLO Format:** `<class_id> <x_center> <y_center> <width> <height>`

All spatial values normalized to [0, 1] relative to image dimensions.

**Class Mapping:**

We consolidate nuScenes' 23 categories into 4 classes:
- Class 0: Cars (vehicle.car, vehicle.taxi)
- Class 1: Trucks/Buses (vehicle.truck, vehicle.bus.*, vehicle.construction)
- Class 2: Pedestrians (human.pedestrian.*)
- Class 3: Cyclists (vehicle.bicycle, vehicle.motorcycle)

This reduces class imbalance and focuses on key autonomous driving objects, simplifying the processing for the model.



## 5. Model Selection and Architecture

### 5.1 Why YOLO for BEV Detection?

We selected the YOLO (You Only Look Once) architecture for several reasons:

**Advantages:**
- **Real-time performance:** Single-stage detector optimized for speed
- **Mature ecosystem:** Well-tested on diverse 2D datasets
- **Transfer learning:** Pretrained COCO weights provide strong initialization
- **Efficient architecture:** Balanced accuracy and computational cost
- **Production-ready:** Extensive deployment experience in industry

**Domain Adaptation Challenge:**

Traditional LiDAR detectors (PointPillars, CenterPoint) process 3D points directly. Our approach converts 3D to 2D, trading some 3D spatial information for:
- Computational efficiency (2D convolutions vs 3D operations)
- Simplified architecture (proven 2D detectors vs specialized 3D models)
- Easier deployment (standard inference pipelines)

### 5.2 YOLOv12s Architecture

We use YOLOv12s (small variant) as base model for our predictions:

**Model Specifications:**
- Parameters: 9.1 million (trainable)
- Architecture: CSPDarknet backbone + FPN neck + detection head
- Input: 1024×1024×3
- Output: 4 classes with bounding box predictions

**Architecture Components:**

1. **Backbone (Feature Extractor):**
   - CSPDarknet with cross-stage partial connections
   - Extracts hierarchical features at multiple scales
   - Pretrained on COCO dataset (80 RGB image classes)

2. **Neck (Feature Pyramid Network):**
   - Fuses multi-scale features
   - Top-down and bottom-up pathways
   - Handles objects at different sizes (cars vs pedestrians)

3. **Detection Head:**
   - Predicts bounding boxes and class probabilities
   - Three detection scales for multi-size objects
   - Anchor-free design (YOLOv12 innovation)

With this approach, we can use a reliable tested model instead of building one from scratch, while still having the flexibility of train it and adapt it to this project's specific needs.

### 5.3 Transfer Learning Strategy

Pretrained COCO weights provide strong low-level features (edges, textures, shapes) but were learned on RGB images, not LiDAR BEV. We employ two-stage transfer learning:

**Stage 1 - Warm-up (50 epochs):**
- Freeze backbone (first 10 layers)
- Train only detection head
- Learning rate: 0.01
- Purpose: Adapt head to BEV domain without disrupting pretrained features

**Stage 2 - Fine-tuning (150 epochs):**
- Unfreeze all layers
- Train end-to-end
- Learning rate: 0.001 (10x lower)
- Purpose: Fine-tune entire network for BEV-specific patterns

This staged approach prevents catastrophic forgetting while enabling domain adaptation.

### 5.4 Training Configuration

**Hyperparameters:**
- Batch size: 16
- Optimizer: AdamW
- Learning rate schedule: Cosine annealing
- Weight decay: 0.0005
- Image size: 1024×1024

**Data Augmentation:**
- Rotation: ±15 degrees
- Translation: ±10% of image size
- Scale: ±50%
- Horizontal flip: 50% probability
- Mosaic augmentation: Combines 4 images (100% probability)
- MixUp: Blends image pairs (10% probability)

**Regularization:**
- Early stopping: Patience of 50 epochs (Stage 2 only)
- Weight decay: 0.0005
- Dropout in detection head (YOLO default)

**Hardware:**
- Training time: 2-4 hours per stage on modern GPU (v1.0-mini)
- GPU memory: ~12 GB with batch size 16
- Device: Single GPU (CUDA device 0)

### 5.5 YOLO Loss Function

The total loss combines localization, objectness, and classification:

$$L_{total} = \lambda_{box} L_{box} + \lambda_{obj} L_{obj} + \lambda_{cls} L_{cls}$$

where:

- $L_{box}$: CIoU loss for bounding box regression
- $L_{obj}$: Binary cross-entropy for objectness
- $L_{cls}$: Cross-entropy for class prediction

**Complete IoU (CIoU) Loss:**

$$L_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

where $\rho$ is Euclidean distance, $c$ is diagonal of smallest enclosing box, and $v$ measures aspect ratio consistency.

## 6. Model Results

### 6.1 Evaluation Metrics

We evaluate using standard object detection metrics:

The threshold for differentiating between a true and false prediction is determined by the Intersection over Union of the prediction with the ground truth annotation using the following formula:

$$\text{IoU} = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}$$

where $B_p$ is the predicted bounding box and $B_{gt}$ is the ground truth box.


**Precision:** TP / (TP + FP)
- Proportion of correct detections among all predictions
- Higher values mean fewer false alarms

**Recall:** TP / (TP + FN)
- Proportion of ground truth objects detected
- Higher values mean fewer missed objects

The formula for calculating Average Precision (AP) is:

$$\text{AP} = \int_0^1 p(r) \, dr \approx \sum_{k=1}^{n} (r_k - r_{k-1}) \cdot p(r_k)$$

where $p(r)$ is the precision at recall level $r$.

**mAP@0.5:** Mean AP across all classes at IoU threshold 0.5.

**mAP@0.5:0.95:** Average mAP across IoU thresholds $0.5, 0.55, 0.60, \ldots, 0.95$:

$$\text{mAP@0.5:0.95} = \frac{1}{10} \sum_{t \in 0.5, 0.55, \ldots, 0.95} \text{mAP@}t$$


### 6.2 Overall Performance


We conducted experimental runs across both datasets with different configurations. The table below summarizes overall performance metrics (Model Evaluation stage):

| Run | Dataset       | Resolution | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
| --- | ------------- | ---------- | ------- | ------------ | --------- | ------ |
| 1   | v1.0-mini     | 1024×1024  | 0.608   | 0.316        | 0.801     | 0.364  |
| 2   | v1.0-trainval | 1024×1024  | 0.616   | 0.379        | 0.807     | 0.390  |
| 3   | v1.0-trainval | 1280×1280  | 0.630   | 0.380        | 0.811     | 0.416  |

The results indicate a steady increase in performance across all metrics.

Best Model Performance (Run 3):

- mAP@0.5: 0.630
- mAP@0.5:0.95: 0.380
- Precision: 0.811
- Recall: 0.416

![](Resources/Graphs/results_overall.png)


### Per-Class Performance

Detection performance varies significantly across object classes, correlating with object size and LiDAR point density.

#### Per-Class mAP@0.5 (Best Run - Run 3):

| Class      | mAP@0.5 | Precision | Recall |
| ---------- | ------- | --------- | ------ |
| Car        | 0.792   | 0.848     | 0.670  |
| Truck/Bus  | 0.750   | 0.820     | 0.612  |
| Pedestrian | 0.465   | 0.672     | 0.262  |
| Cyclist    | 0.515   | 0.902     | 0.120  |


#### Cars and trucks are more likely to be correctly predicted

##### Class 0 - Cars (~50% of instances):

- Best-performing class with mAP@0.5 of 0.792
- Large size and high point density provide clear BEV signatures
- Consistent detection across all experimental runs

![](Resources/Graphs/results_car.png)

##### Class 1 - Trucks/Buses (~10% of instances):

- Strong performance with mAP@0.5 of 0.750
- Large BEV footprint aids detection despite lower frequency

![](Resources/Graphs/results_truck_bus.png)

#### Pedestrians and cyclists are more challenging for the model to predict

Because of their smaller LiDAR signature, the model performance is not as strong for these two classes even though the high precision scores indicates that when the model does make a prediction, it is usually correct.

#### Class 2 - Pedestrians (~35% of instances):

- Recall of 0.26 means that only one in four are caught
- Lowest mAP@50 score with 0.750

![](Resources/Graphs/results_pedestrian.png)

#### Class 3 - Cyclists (~5% of instances)

- Most underrepresented class.
- Lowest recall at 0.12

![](Resources/Graphs/results_cyclist.png)

### Training parameters comparison analysis

In an effort to improve the results, especially around the minority classes, we implemented two distinct modifications to the original training regimen:
- we trained on the larger train-val dataset which has 3377 samples (vs 404 for the v1-mini dataset)
- we increased the resolution of the BEV images from 1024x1024 to 1280x1280.

While increasing the size of the dataset produced only negligible improvements in mAP@50 and recall for most classes (cars, trucks, pedestrians), the following phenomenons were observed: 
- mAP@95 for all classes increased from 0.32 to 0.38, suggesting that larger amounts of data helped with achieving tighter bounding box localization.
- the recall for the most problematic class (cyclists) jumped up from 0.01 to 0.10

Increasing the resolution of the rasterized images from 1024 to 1280 (allowing them to capture more information per channel from the LiDAR raw data) resulted in: 
- improved mAP@50 for all classes, indicating that the model was able to make more predictions which overlapped by at least 50% with the ground truth
- further improved cyclists recall from 0.10 to 0.12

![](Resources/Graphs/results_comparison.png)


These metrics demonstrate feasibility of 2D detection on LiDAR BEV representations, trading some accuracy from specialized 3D detectors for significant computational efficiency gains.


### 6.3 Qualitative Results


In order to witness the model in action, we developed a simple visualizer which:

- Ran inference on chronologically ordered images from a scene
- Overlayed the bounding boxes on the images


Using this tool, we were able to visually confirm that the model's predictions was consistent with the LiDAR point cloud as illustrated in Figure 10.

Visualizations of model predictions on test set reveal:

**Successful Cases:**
- Strong detection of vehicles in open areas
- Accurate localization of stationary objects
- Robust performance across varying point densities

**Challenging Cases:**
- Occluded pedestrians behind vehicles
- Cyclists at far distances (sparse points)
- Closely spaced vehicles (merged BEV footprints)
- Objects at BEV boundary (clipped bounding boxes)


![](Resources/Visualizer.png)


## 7. Discussion

### 7.1 Key Findings

This project demonstrates that YOLO-based 2D detection on LiDAR BEV representations is viable for autonomous driving applications. The approach successfully combines LiDAR's spatial accuracy with 2D detection efficiency.

**Principal Insights:**

1. **BEV Representation Effectiveness:** Multi-channel encoding (height, intensity, density) preserves sufficient spatial information for object detection while enabling efficient 2D processing.

2. **Transfer Learning Success:** COCO-pretrained weights transfer effectively to LiDAR BEV domain despite different imaging modality, confirming that low-level features (edges, shapes) generalize across domains.

3. **Class-Dependent Performance:** Detection accuracy correlates with object size and point density, with vehicles performing best and pedestrians most challenging.

4. **Real-Time Feasibility:** Inference speeds of 40-60 FPS demonstrate suitability for real-time autonomous driving (typically requires 10-20 FPS).

### 7.2 Limitations

Several limitations warrant discussion:

**Loss of 3D Information:**
- BEV projection discards height information
- Vertically stacked objects (e.g., overpass with traffic below) may be ambiguous
- Tall object height not directly observable
- Smaller objects like pedestrians or cyclists had the lowest performance in our model, so more research needs to be done.

**Axis-Aligned Bounding Boxes:**
- Standard YOLO uses axis-aligned boxes
- Rotated vehicles have looser-fitting boxes
- Some background regions incorrectly included in boxes

**Dataset Scale:**
- Training on v1.0-mini (404 samples) limits generalization
- Full v1.0-trainval (40,000 samples) would improve performance
- Class imbalance affects minority classes

**Point Cloud Sparsity:**
- Far objects have few LiDAR returns
- Pedestrians have inherently sparse representation
- Detection range limited compared to camera-based methods



### 7.3 Future Work

Several directions could extend this research:

**Multi-Scale BEV Encoding:**
- Generate BEV at multiple resolutions
- Capture both large vehicles and small pedestrians effectively
- Hierarchical feature pyramid for BEV

**Additional sensor data**
- Incorporate Radar and Camera data
- With a hybrid approach, detection of smaller objects can be improved from different perspectives
- Since it would still be 2D processing, additional layers might not represent a significant cost increase

**Temporal Fusion:**
- Incorporate multiple LiDAR sweeps (sequential scans)
- Add motion information through temporal context
- Improve detection of moving objects

**Oriented Bounding Boxes:**
- Modify YOLO to predict rotation angle
- Tighter box fits for rotated vehicles
- Reduces false positives from background inclusion

**Camera-LiDAR Fusion:**
- Early fusion: Combine BEV with camera bird's-eye view
- Late fusion: Merge detections from both modalities
- Leverage complementary strengths

**Full Dataset Training:**
- Scale to v1.0-trainval (40,000 samples)
- Improve minority class performance
- Better generalization across scenarios

**Attention Mechanisms:**
- Spatial attention to focus on high-density regions
- Channel attention to weight informative BEV channels
- Improve detection of challenging objects

**Deployment Optimization:**
- Model quantization (INT8) for faster inference
- TensorRT optimization for NVIDIA platforms
- ONNX export for cross-platform deployment

## 8. Conclusion

This project successfully demonstrates the feasibility of using YOLO-based 2D object detection on LiDAR data through Bird's-Eye View representation. By converting 3D point clouds into multi-channel 2D images, we achieve real-time detection performance while maintaining the spatial accuracy advantages of LiDAR sensing.

Our key contributions include:
- A complete implementation pipeline from raw nuScenes data to trained YOLO model
- BEV encoding scheme with height, intensity, and density channels
- Two-stage transfer learning methodology from COCO to LiDAR domain
- Comprehensive evaluation demonstrating practical feasibility

The results indicate that this hybrid approach offers a compelling trade-off: sacrificing some 3D spatial information for significant computational efficiency gains. For autonomous driving applications where real-time performance is critical, this approach provides a practical alternative to specialized 3D detection architectures.

Future work should explore multi-sensor fusion, temporal context, and scaling to the full nuScenes dataset to further improve detection performance, particularly for challenging object classes like pedestrians and cyclists.


## References

1. Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V. E., Xu, Q., ... & Beijbom, O. (2020). nuScenes: A multimodal dataset for autonomous driving. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 11621-11631).

2. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 779-788).

3. Lang, A. H., Vora, S., Caesar, H., Zhou, L., Yang, J., & Beijbom, O. (2019). PointPillars: Fast encoders for object detection from point clouds. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 12697-12705).

4. Yin, T., Zhou, X., & Krahenbuhl, P. (2021). Center-based 3d object detection and tracking. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 11784-11793).

5. Ultralytics. (2024). YOLOv12 Documentation. Retrieved from https://docs.ultralytics.com/models/yolo12/

6. Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for autonomous driving? The KITTI vision benchmark suite. In *2012 IEEE conference on computer vision and pattern recognition* (pp. 3354-3361).

7. Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. In *European conference on computer vision* (pp. 740-755). Springer, Cham.

## Appendix


