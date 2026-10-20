<div align="center">

# THERMOEDGEVISION - Smart Cooling Edge AI

**Intelligent Real-Time Classroom Occupancy Detection & Automated AC Energy Optimization at the Edge**

[![Python](https://img.shields.io/badge/Python-3.10.19-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA_CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Ultralytics YOLOv11](https://img.shields.io/badge/YOLOv11-8.4.126-purple?style=for-the-badge)](https://ultralytics.com/)
[![ONNX Runtime GPU](https://img.shields.io/badge/ONNX_Runtime-1.23.2-orange?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO_Tuning-blueviolet?style=for-the-badge)](https://optuna.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.2-lightgrey?style=for-the-badge&logo=flask&logoColor=black)](https://flask.palletsprojects.com/)
[![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Atlas_Cloud-darkgreen?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## Executive Summary & System Overview

**Smart Cooling Edge AI (THERMOEDGEVISION)** is an enterprise-grade Edge AI solution engineered to optimize Air Conditioning (AC) energy consumption across university classrooms and computer laboratories.

Instead of running central AC units at full blast in empty or partially occupied rooms, the system ingests live CCTV camera streams, executes real-time student detection using a multithreaded **YOLOv11 ONNX Runtime GPU Acceleration Engine**, and dynamically regulates AC climate settings (active AC units, target temperatures, fan speeds, louver positions) based on live student occupancy. All occupancy metrics, AC state changes, and annotated frame snapshots are asynchronously logged to **MongoDB Atlas Cloud**, establishing a **continuous data flywheel** designed to transition the current rule-based climate engine into a **fully predictive Machine Learning / Reinforcement Learning climate control model** in future iterations.

---

## Key System Features

- **Asynchronous Producer-Consumer Threading:** Completely decouples video frame rendering (**30+ FPS**) from compute-bound YOLO inference and cloud IO.
- **Adaptive CLAHE Brightness Enhancer:** Automatically applies LAB color space Contrast Limited Adaptive Histogram Equalization for shadowed or low-light classroom vision (`brightness < 80`).
- **Vectorized NumPy NMS Post-Processing:** Replaces slow Python loops with matrix indexing (`preds[:, 4:]`) over 8,400 detection anchors for 10x-100x faster post-processing.
- **2-Stage MLOps Pipeline:** Integrated **Optuna Hyperparameter Optimization (HPO)** and full 200-epoch retraining with experiment tracking via **MLflow**.
- **Dynamic AC Climate Regulation Engine:** Matrix-based rule engine dynamically controls up to 4 AC units according to 5 occupancy brackets defined in `params.yaml`.
- **Hardware-Agnostic Dual Execution Engine:** Engineered for high-speed NVIDIA CUDA GPU acceleration (`CUDAExecutionProvider`) to achieve maximum performance and minimum latency, while maintaining 100% hardware-agnostic compatibility to automatically fall back to CPU execution (`CPUExecutionProvider`) on systems without dedicated GPUs.
- **Cloud Telemetry & Data Flywheel (Future ML Transition):** Asynchronously streams structured occupancy metrics, AC state changes, and image captures to MongoDB Atlas Cloud over TLS SSL, accumulating historical training data to replace the rule-based engine with predictive ML cooling control models.

---

## Modular System Architecture

The production software is built as a clean, decoupled Python package (`Smart_cooling_edge_AI`) separating IO-bound video streaming from compute-bound AI inference and network logging:

```text
Smart-cooling-edge_AI/
├── Smart_cooling_edge_AI/                # Production Modular Package
│   ├── pipeline/                         # Multithreaded Pipeline Architecture
│   │   ├── prediction_pipeline.py        # Main Thread Frame Display & Web Stream Orchestrator
│   │   └── inference_thread.py           # Daemon Worker Thread & Async Logging Executor
│   ├── components/                       # Core Business & Inference Logic
│   │   ├── model_predictor.py            # ONNX Runtime Session & Vectorized NumPy NMS
│   │   └── cooling_controller.py         # Occupancy Rule Engine & Climate Controller
│   ├── dbdata_access/                    # Database Operations
│   │   └── data_logger.py                # Async MongoDB Atlas Cloud Telemetry Logger
│   ├── logging/                          # System Execution Logger
│   │   └── logger.py                     # Centralized Dual-Handler File & Console Logger
│   └── utils/                            # Utilities & Preprocessing
│       ├── image_utils.py                # CLAHE Enhancement, Letterboxing & Annotations
│       └── configurations.py             # YAML Parameter Reader
├── models/                               # Exported Production ONNX Model (yolo11n.onnx - 10.1 MB)
├── notebooks/                            # MLOps Experiments & HPO Tuning
│   ├── train_yolo11.ipynb                # Optuna HPO & 2-Stage Retraining Pipeline
│   ├── experiment.ipynb                  # Edge AI Research & Prototyping Experiments
│   ├── mlflow.db                         # Local SQLite MLflow Experiment Database
│   └── hpo_experiments/                  # HPO Trials & Retrained Best Model Artifacts
│       ├── HPO_Trials/                   # Optuna Trial Runs (optuna_trial_0 to 4)
│       └── Final_Best_Model/             # 200-Epoch Retrained Model Weights & Plots
├── raw_data/                             # Raw Training Data & Frame Ingestion
│   ├── videos/                           # Input Raw Classroom Videos
│   └── raw_frames/                       # Extracted Frames for Roboflow Annotations
├── templates/                            # Web UI Templates (dashboard.html, settings.html, etc.)
├── static/                               # Web UI Static Assets (CSS, JS, Icons, Images)
├── Frame_captures/                       # Saved Retraining Frame Snapshots
├── temp_uploads/                         # Temporary Video Upload Storage
├── logs/                                 # System Execution Logs (system_execution.log)
├── app.py                                # Main Flask Web Backend Server
├── demo.py                               # Standalone Prediction Pipeline Demo Runner
├── extract_frames.py                     # Automated Video Frame Sampling Utility
├── template.py                           # Automated Project Directory Structure Generator
├── setup.py                              # Setuptools Package Registration Script
├── requirements.txt                      # Project Dependencies
├── params.yaml                           # System Configuration Parameters
├── Dockerfile                            # Docker Containerization Specification
├── .dockerignore                         # Docker Build Ignore Rules
├── .env                                  # Environment Variables & Secret Credentials
├── LICENSE                               # License File
└── README.md                             # Comprehensive Project Documentation
```

---

## Detailed Component Code Audit

### 1. Multithreaded Pipeline Architecture (`Smart_cooling_edge_AI/pipeline/`)
- **`prediction_pipeline.py` (`PredictionPipeline` Class):**
  - Reads `PIPELINE_CONFIG` parameters (`video_source`, `interval_seconds`) safely from `params.yaml`.
  - Instantiates `CoolingController` and `InferenceWorker` daemon worker thread, launching it immediately on startup (`self._worker.start()`).
  - `process_frame(frame)`: Receives raw frames from Flask, pushes non-blocking copies to `InferenceWorker` via `set_frame()`, fetches latest detection coordinates via `get_results()`, overlays green bounding boxes and red label badges (`draw_boxes`), and updates the shared display frame cache (`_current_display_frame`) under a thread lock (`_frame_lock`).
  - `get_current_display_frame()`: Provides thread-safe access to display frame copies for MJPEG web streaming.
  - `run_pipeline()`: Standalone local execution loop for OpenCV display windows with keyboard interrupt exit (`q`).
  - `stop()`: Signals the worker thread to exit cleanly (`_worker.stop()`).

- **`inference_thread.py` (`InferenceWorker` Class):**
  - Runs continuous ONNX inference in a background daemon thread (`_worker_loop`) to ensure the main display loop is never blocked by model execution.
  - Implements internal thread synchronization primitives (`_frame_lock` and `_result_lock`).
  - Measures per-frame inference latency (`latest_latency_ms`).
  - **Dual-Timing Operational Architecture:**
    - **Real-Time Student Detection (Every Frame / 30+ FPS):** YOLO ONNX detection and bounding box coordinate updates execute continuously on every camera frame in real time.
    - **5-Minute Throttled HVAC Regulation & Cloud Logging (Every 300 Seconds):** Physical AC climate regulation (`regulate_cooling_system`), image capture saving (`Frame_captures/`), and MongoDB cloud logging are throttled to execute every **5 minutes (300 seconds)** (`interval_seconds`).
    - **HVAC Engineering Rationale:** Prevents damaging AC compressor cycling (rapid ON/OFF fluctuation) when students temporarily move or walk past camera view, while matching the natural thermal inertia of large classroom environments. Offloads cloud logging to a secondary **`ThreadPoolExecutor(max_workers=2)`** so network IO never stalls AI inference.

### 2. Core Business & Inference Logic (`Smart_cooling_edge_AI/components/`)
- **`model_predictor.py` (`ModelPredictor` Module):**
  - Performs automatic Windows CUDA DLL directory registration (`os.add_dll_directory`) for NVIDIA GPU runtime execution.
  - Initializes ONNX Runtime Session using dual execution providers `providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']`. If a CUDA GPU runtime is detected, it runs with high-speed GPU acceleration; otherwise, it gracefully falls back to CPU execution without requiring any code or environment modifications.
  - `run_detection(frame)`:
    - Preprocesses input frame into letterboxed CHW normalized float32 tensor (1, 3, 640, 640).
    - Executes ONNX forward pass (`session.run`), squeezing outputs to prediction matrix shape (8400, 84).
    - **Vectorized NumPy NMS Parsing:** Extracts class scores (`preds[:, 4:]`), computes max confidence scores, and filters for Person class (`class_id == 0`) exceeding `CONFIDENCE_THRESHOLD` using parallel array masking (`mask = (class_ids == 0) & (max_scores >= CONFIDENCE_THRESHOLD)`) for 10x-100x faster execution than Python loops.
    - Applies `cv2.dnn.NMSBoxes` filtering with `NMS_THRESHOLD`.
  - `update_thresholds(new_conf, new_nms)`: Dynamically updates in-memory confidence and NMS thresholds instantly without reloading model weights.

- **`cooling_controller.py` (`CoolingController` Class):**
  - Dynamically loads AC automation rules from `params.yaml` (`AC_CONTROL_RULES`).
  - `regulate_cooling_system(student_count)`: Matches live student count against defined min/max range brackets.
  - Returns structured HVAC control dictionary containing active AC count (1-4), AC mode, target temperature min/max (17°C - 26°C), fan speed, louver position, and description.
  - Includes a safe hardcoded fallback state (`_get_default_fallback_state()`) to prevent system crashes during unexpected exceptions.

### 3. Database Access & System Logging (`Smart_cooling_edge_AI/dbdata_access/` & `logging/`)
- **`data_logger.py` (`db_system_logging()` Function):**
  - Establishes TLS SSL connection to MongoDB Atlas Cloud using `certifi` CA certificates (`MongoClient(CONNECTION_URI, tlsCAFile=certifi.where())`).
  - Asynchronously inserts structured JSON documents containing `timestamp`, `student_count`, `occupancy_level`, `hvac_control` details, and `image_path`.
  - **Strategic MLOps Purpose:** Accumulates real-world historical occupancy logs and climate response telemetry in MongoDB Atlas Cloud to construct a training dataset for replacing the initial rule-based cooling system (`cooling_controller.py`) with a predictive Machine Learning cooling control model in future releases.
  - Fails gracefully without crashing the main AI pipeline if database is offline or unconfigured.

- **`logger.py` (`get_logger()` Function):**
  - Provides a centralized logger writing simultaneously to stdout console, timestamped log files (`logs/log_YYYY-MM-DD_HH-MM-SS.log`), and a continuous main log file (`logs/system_execution.log`).

### 4. Image Utilities & YAML Reader (`Smart_cooling_edge_AI/utils/`)
- **`image_utils.py`:**
  - `preprocess(frame)`: Converts BGR to RGB. Checks average gray level (`np.mean(frame_gray)`). If `brightness < 80` (shadowed/dark classroom), automatically applies LAB color space CLAHE contrast enhancement (`clipLimit=2.0, tileGridSize=(8,8)`). Performs letterbox resizing (640x640) with constant padding (114, 114, 114), normalizes [0, 1], transposes to CHW tensor, and adds batch dimension.
  - `draw_boxes(frame, boxes)`: Draws green bounding boxes (`0, 255, 0`), red label background badges (`0, 0, 255`), and white text labels `student: 0.85`.
  - `save_annotated_frame(...)`: Overlays summary info (timestamp, student count, target temp) and saves captured frame snapshot to `Frame_captures/` for future model retraining.

- **`configurations.py`:**
  - `read_yaml(path_to_yaml)`: Reads and parses YAML configuration files with UTF-8 encoding and exception handling.

### 5. Flask Web Server & Dashboard (`app.py`)
- **Web UI Routes:** `/` (10s loading screen), `/dashboard` (Control dashboard with turbine animations), `/nodes` (Hardware telemetry matrix), `/analytics` (Live log terminal & MongoDB documents), `/settings` (Dynamic threshold sliders with HPO badges).
- **REST APIs:** `/api/upload_video` (video file upload with auto-cleanup of previous temp files via `cleanup_temp_file`), `/api/stop_stream`, `/api/status` (telemetry API returning CPU, GPU, RAM, VRAM, uptime, latency, and AC state), `/api/analytics_logs`, `/api/get_model_config`, `/api/update_model_config`.
- **`gen_frames()` Stream Generator:**
  - Renders a cyan Standby screen ("THERMOEDGEVISION - STANDBY MODE") when no video source is loaded.
  - Preserves playback position across page navigation (`pipeline.current_frame_pos`).
  - Loops seamlessly back to frame 0 when video reaches EOF.

---

## System Architecture Diagram

<div align="center">

[![System Architecture](./resources/Diagram/THERMOEDGEVISION%20smart%20coolin%20ai.jpg)](./resources/Diagram/THERMOEDGEVISION%20smart%20coolin%20ai.jpg)

> *Click the diagram image above to open the full high-resolution architecture visual.*

</div>

---

---

## End-to-End MLOps Lifecycle & Modular Transition Strategy

The project implements a complete, **closed-loop telemetry feedback MLOps data flywheel architecture** that seamlessly bridges initial data ingestion and research prototyping with production modular deployment, real-world actuation, and semi-automated retraining:

```text
┌───────────────────────────────┐     ┌───────────────────────────────────┐     ┌───────────────────────────────────┐
│ 1. Data Ingestion & Sampling  │ ──► │ 2. Research & Vision Prototyping  │ ──► │ 3. Optuna HPO & MLflow Tracking   │
│   (extract_frames.py)         │     │    (notebooks/experiment.ipynb)   │     │    (notebooks/train_yolo11.ipynb)  │
└───────────────────────────────┘     └───────────────────────────────────┘     └─────────────────┬─────────────────┘
                                                                                                  │
┌───────────────────────────────┐     ┌───────────────────────────────────┐                       │
│ 5. Continuous Data Flywheel   │ ◄── │ 4. Production Modular Engine      │ ◄─────────────────────┘
│    (MongoDB Atlas & Captures) │     │    (Smart_cooling_edge_AI)        │      (ONNX Slimming Export: models/yolo11n.onnx)
└───────────────────────────────┘     └───────────────────────────────────┘
```

### 1. Data Ingestion & Frame Sampling (`extract_frames.py`)
Classroom video feeds are systematically sampled (`FRAME_INTERVAL = 300`) to extract raw images (`raw_data/raw_frames/`) for Roboflow student bounding box annotations.

### 2. Vision Algorithm Prototyping (`notebooks/experiment.ipynb`)
Core computer vision algorithms are prototyped and benchmarked:
- **ONNX Runtime Prototyping:** Evaluates performance gains of ONNX Runtime vs native PyTorch inference for lightweight edge deployment.
- **Execution Provider Benchmarking:** Benchmarks `CUDAExecutionProvider` against `CPUExecutionProvider` to verify GPU acceleration speedup.
- **CLAHE Contrast Enhancement Prototyping:** Tests LAB color space Contrast Limited Adaptive Histogram Equalization for low-light classroom vision (`brightness < 80`).
- **Vectorized NumPy NMS Prototyping:** Implements matrix indexing (`preds[:, 4:]`) across 8,400 raw detection anchors for 10x-100x faster post-processing.

### 3. Optuna HPO & MLflow Experiment Tracking (`notebooks/train_yolo11.ipynb`)
- **Stage 1 — Optuna HPO Study (5 Trials):** Systematically samples 9 hyperparameters (`lr0`, `lrf`, `momentum`, `weight_decay`, `box`, `cls`, `hsv_h`, `hsv_s`, `hsv_v`) across 5 trials (30 epochs each). Logs metrics (`mAP50`, `mAP50-95`, `precision`, `recall`), evaluation plots, and model weights to a SQLite MLflow database (`sqlite:///mlflow.db`).

#### Launching the MLflow Interactive UI Dashboard:
To inspect, compare, and analyze all Optuna trial runs, hyperparameter parallel coordinates, and loss progression graphs in your browser:

```bash
# Open terminal in project root and launch MLflow server
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open your browser at: **`http://127.0.0.1:5000`** (or `http://localhost:5000`)

### 4. Winning Hyperparameter Selection & Impact Analysis
- **Optuna HPO Winner Selection:** Across the 5 HPO trials (`optuna_trial_0` to `4`), **Trial 1 achieved the highest validation mAP@50 score of 83.71% (0.8371)**. The winning hyperparameter set (`winning_params = study.best_params`) identified was:

#### Hyperparameter Function & Real-World Training Impact Matrix:

| Hyperparameter | Optuna Value | Optimization Role | Real-World Model Impact |
| :--- | :---: | :--- | :--- |
| **`lr0`** (Initial Learning Rate) | `0.0007107` | Initial gradient descent step size | Prevents gradient explosion; ensures smooth loss optimization on fine-tuning datasets |
| **`lrf`** (Final LR Factor) | `0.0460` | Cosine annealing LR decay floor | Decays LR to ~3.2e-5 for fine-grained weight convergence in late training epochs |
| **`momentum`** | `0.9673` | SGD gradient direction accumulator | Dampens oscillations in steep loss valleys, accelerating gradient convergence |
| **`weight_decay`** | `0.000583` | L2 weight regularization penalty | Prevents neural network weights from growing excessively large, mitigating overfitting |
| **`box`** (Box Loss Gain) | `6.1151` | Bounding box CIoU loss multiplier | Forces model to heavily prioritize precise student boundary & coordinate localization |
| **`cls`** (Class Loss Gain) | `1.4571` | Classification BCE loss multiplier | Calibrates student confidence scores against bounding box localization accuracy |
| **`hsv_h`** (Hue Augmentation) | `0.0142` | Random image hue shift (±1.4%) | Robustness against classroom lighting color temperature shifts (warm vs cool LED) |
| **`hsv_s`** (Saturation Aug) | `0.2432` | Random color saturation shift (±24.3%) | Generalization across diverse student clothing colors and camera color profiles |
| **`hsv_v`** (Brightness Aug) | `0.5150` | Random image brightness shift (±51.5%) | Ensures student detection in shadowed classroom corners and high-glare windows |

- **Stage 2 — Full 200-Epoch Retraining:** Retraining YOLOv11 from scratch using `winning_params` boosted validation **mAP@50 from 83.71% up to 88.4% (0.884)**, achieving optimal convergence at epoch 141 (early stopping patience = 50). Outputs were saved to `notebooks/hpo_experiments/Final_Best_Model/`.
- **Stage 3 — ONNX Export & Slimming:** PyTorch weights were exported to ONNX format with ONNX Slimming (`imgsz=640`, `simplify=True`), creating a lightweight **10.1 MB** production model (`models/yolo11n.onnx`).
- **Modular Packaging Transition:** Transitioned research notebook code into a clean, decoupled production Python package (`Smart_cooling_edge_AI/`) registered via `setup.py` (`pip install -e .`).

### 5. Empirical Derivation of Production Thresholds (`conf = 0.45`, `nms = 0.60`)
- **Confidence Threshold Derivation (`confidence_threshold = 0.45`):**
  - Derived mathematically from the **F1-Confidence Curve (`BoxF1_curve.png`)**.
  - The F1 curve shows a maximum peak F1-score of **0.87 (87.0%) at `conf = 0.46`**.
  - Setting the production default to **`0.45`** guarantees an optimal balance between precision (85.3%) and recall (87.2%), preventing false alarms while ensuring maximum detection sensitivity for partially occluded students.
- **NMS IoU Threshold Derivation (`nms_threshold = 0.60`):**
  - Derived empirically from **IoU Overlap Validation Sweeps** on classroom computer lab CCTV feeds.
  - Students seated at adjacent computer lab desks create bounding box overlaps ranging between 50% to 60% IoU.
  - Setting the default NMS threshold to **`0.60`** prevents adjacent students from being falsely suppressed as duplicate detections, while effectively eliminating true redundant overlapping boxes for the same individual.

### 6. Production Multithreading & Data Flywheel (Semi-Closed Loop Strategy)
- **Asynchronous Edge Runtime:** Production pipeline decouples display rendering (`prediction_pipeline.py`) from AI inference daemon threads (`inference_thread.py`) and offloads MongoDB cloud logging to a `ThreadPoolExecutor(max_workers=2)`.
- **Dual-Timing Operational Execution:** Detection operates in **real time on every frame (30+ FPS)** for visual overlay, while physical AC climate actuation, image capture archiving (`Frame_captures/`), and MongoDB cloud logging are throttled to run **every 5 minutes (300s)** to protect AC compressors against rapid ON/OFF fluctuation damage and respect room thermal inertia.
- **Strategic Continuous Data Flywheel:** Annotated frame captures are saved locally to `Frame_captures/` while live telemetry (occupancy count, active ACs, target temp, fan speed, louver) is streamed to MongoDB Atlas Cloud.
- **Semi-Automated Retraining Feedback Loop:** Operational telemetry and image capture logging are 100% automated at runtime. Periodically, accumulated real-world data in `Frame_captures/` is sampled, annotated, and passed to `notebooks/train_yolo11.ipynb` for Optuna HPO retraining, exporting an updated `models/yolo11n.onnx` model and completing the semi-closed MLOps feedback loop.

### 7. Engineering & Hardware Memory Optimizations
- **GTX 1650 Target Hardware Optimization:** Tailored specifically for the host training hardware (**NVIDIA GeForce GTX 1650 4GB VRAM**). At 640x640 resolution, standard `batch=16` training exceeds 4GB VRAM (~4.18 GB), triggering CUDA Out-Of-Memory (OOM) crashes. Setting `batch=8` and `workers=4` stabilizes GPU VRAM consumption at **~2.25 GB**, guaranteeing 100% training stability.
- **CUDA Cache Clearing Protocol:** Explicit garbage collection (`gc.collect()`) and GPU memory cache clearing (`torch.cuda.empty_cache()`) are executed at the start and end of every trial to prevent memory leaks across runs.

---

## Empirical Model Performance Benchmark & Evaluation

Evaluated on 337 validation student instances across classroom CCTV test feeds:

| Metric | Score | Analytical Insight |
| :--- | :--- | :--- |
| **mAP@50** | **88.4% (0.884)** | Exceptional overall bounding box detection accuracy |
| **Peak F1-Score** | **0.87 (87.0%)** | Achieved at **`conf = 0.46`** optimal balance point |
| **Max Detection Recall** | **93.0% (0.930)** | 313 out of 337 student instances correctly detected |
| **Max Detection Precision**| **100.0% (1.000)** | Reached at **`conf = 0.80`** with zero false alarms |
| **Overfitting Status** | **Zero Divergence** | Train and validation loss curves converged smoothly in parallel |
| **Production Thresholds** | **`conf = 0.45`**, **`nms = 0.60`** | Synchronized across `params.yaml`, Backend, and Web Dashboard |

<div align="center">

### Evaluation Curves & Benchmark Plots Gallery

#### 1. Training & Validation Loss & Metric Progression (`results.png`)
<img src="./notebooks/hpo_experiments/Final_Best_Model/results.png" alt="Training & Validation Loss Curves" width="800"/>

> *Smooth parallel convergence of Box Loss, Class Loss, DFL Loss, mAP@50, and mAP@50-95 over 200 training epochs with zero overfitting.*

<br/>

| F1-Confidence Curve (`BoxF1_curve.png`) | Precision-Recall Curve (`BoxPR_curve.png`) |
| :---: | :---: |
| <img src="./notebooks/hpo_experiments/Final_Best_Model/BoxF1_curve.png" alt="F1-Confidence Curve" width="400"/> | <img src="./notebooks/hpo_experiments/Final_Best_Model/BoxPR_curve.png" alt="Precision-Recall Curve" width="400"/> |
| *Peak F1-Score of **0.87 at conf = 0.46** optimal balance point* | *High Area Under Curve (**AUC = 0.884 mAP@50**)* |

<br/>

| Confusion Matrix (`confusion_matrix.png`) | Classroom Validation Detection Output (`val_batch0_pred.jpg`) |
| :---: | :---: |
| <img src="./notebooks/hpo_experiments/Final_Best_Model/confusion_matrix.png" alt="Confusion Matrix" width="400"/> | <img src="./notebooks/hpo_experiments/Final_Best_Model/val_batch0_pred.jpg" alt="Validation Prediction Overlay" width="400"/> |
| *High true-positive rates with minimal background false positives* | *Real-world student bounding box prediction overlay on CCTV feed* |

</div>

---

## AC Automation Rule Matrix

AC units and climate settings adjust dynamically based on student count rules defined in `params.yaml`:

| Occupancy Bracket | Student Count | Active ACs | Mode | Target Temp | Fan Speed | Louver |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Empty** | `0` | `1 Unit` | `Baseline` | `25°C - 26°C` | `Low` | `Down` |
| **Low** | `1 - 20` | `2 Units` | `Alternate` | `24°C - 25°C` | `Low` | `Down` |
| **Medium** | `21 - 60` | `4 Units` | `Normal` | `22°C - 23°C` | `Medium` | `Swing` |
| **High** | `61 - 110` | `4 Units` | `Active` | `20°C - 21°C` | `High` | `Horizontal` |
| **Full / Turbo** | `111 - 150+` | `4 Units` | `Turbo` | `17°C - 19°C` | `Max` | `All Swing` |

---

## Web UI Dashboard Pages & REST APIs

Built with a glassmorphic dark-mode UI using Tailwind CSS and monospaced typography:

- **Landing Page (`/`)**: 10-second dynamic loading screen (`loading.html`).
- **Control Dashboard (`/dashboard`)**: Real-time video player, 4 AC Unit status cards with rotating turbine graphics, target temperature, and edge hardware widget.
- **Edge Nodes Matrix (`/nodes`)**: Real-time CPU/GPU utilization, RAM/VRAM, ONNX execution provider, inference speed, and active subsystem cards.
- **Analytics & DB Logs (`/analytics`)**: Live streamed execution logs terminal and MongoDB Atlas document cards.
- **Settings (`/settings`)**: Dynamic threshold sliders (`conf`, `nms`) with HPO analytics badges and real-time impact descriptions synced with `params.yaml`.

### Primary Flask REST Endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/video_feed` | `GET` | Live annotated MJPEG video stream |
| `/api/upload_video` | `POST` | Upload video file and switch stream source seamlessly |
| `/api/stop_stream` | `POST` | Stop stream and clean temporary video file |
| `/api/status` | `GET` | Hardware telemetry API (CPU, GPU, RAM, VRAM, ONNX, IP) |
| `/api/analytics_logs` | `GET` | Fetch latest telemetry logs from MongoDB Cloud |
| `/api/get_model_config` | `GET` | Fetch active confidence and NMS thresholds |
| `/api/update_model_config` | `POST` | Dynamically update detection thresholds in memory and `params.yaml` |

---

## Enterprise Security, System Resilience & Protection Protocols

The system incorporates production security, safety, and resilience standards across database operations, configuration parsing, and environment variables:

1. **Environment Variable Decoupling & Secret Isolation (`.env` + `python-dotenv`):**
   - Critical credentials (e.g., MongoDB connection URI `MONGODB_URI`, `DB_NAME`, `COLLECTION_NAME`) are decoupled from source code and loaded into memory securely via `python-dotenv`.
   - `.env` files are isolated from source control and deployment artifacts via `.gitignore` and `.dockerignore` to prevent password leaks.

2. **Safe YAML Configuration Parsing (`yaml.safe_load`):**
   - System parameter loading in `configurations.py` uses `yaml.safe_load()` instead of untrusted `yaml.load()`.
   - Protects against arbitrary Python code execution vulnerabilities through malicious YAML configuration payloads.

3. **Graceful Fault Tolerance & Error Isolation Boundaries:**
   - Database operations (`data_logger.py`) and climate control evaluations (`cooling_controller.py`) are protected within `try-except` error isolation boundaries.
   - If cloud databases or network connections drop, the primary AI edge inference and AC regulation loops continue executing safely without crashing.

---

## Installation & Setup Guide

### 1. Prerequisites
- **Python 3.10+**
- **NVIDIA GPU Drivers & CUDA Toolkit 12.1** (Optional — ONNX Runtime automatically falls back to CPU execution if GPU is absent)
- Active **MongoDB Atlas** cluster connection string.

### 2. Virtual Environment Setup
```bash
git clone https://github.com/PasinduGunathilake/Smart-cooling-edge_AI.git
cd Smart-cooling-edge_AI

# Windows
python -m venv torchenv
torchenv\Scripts\activate

# Linux / macOS
python3 -m venv torchenv
source torchenv/bin/activate
```

### 3. Install Package Dependencies
Install the core package in editable mode:
```bash
pip install -e .
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
MONGODB_URI="mongodb+srv://<user>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority"
DB_NAME="smart_cooling_db"
COLLECTION_NAME="AC_System_logs"
```

### 5. Execution Environment Options (Local GPU vs Google Colab)
- **Local GPU (Recommended):** Open `notebooks/train_yolo11.ipynb` in VS Code or Jupyter Notebook and select your PyTorch GPU kernel.
- **Google Colab (Free T4 GPU Alternative):** Upload `notebooks/train_yolo11.ipynb` to [Google Colab](https://colab.research.google.com/), select **Runtime -> Change runtime type -> T4 GPU**, and execute the training pipeline. Download the exported `yolo11n.onnx` file and place it at `models/yolo11n.onnx`.

---

## Running the Application

### Launch Production Web Server
```bash
python app.py
```
Open your browser at: `http://127.0.0.1:5000`

### Launch Standalone OpenCV Demo
```bash
python demo.py
```

### Docker Container Deployment
```bash
# 1. Build Docker Image
docker build -t smart-cooling-edge-ai .

# 2. Run Container
docker run -d -p 5000:5000 --env-file .env --name smart-cooling-app smart-cooling-edge-ai
```

---

## Author

### Pasindu Gunathilake
- **Education**: BSc. Computer Science Undergraduate, University of Jaffna, Sri Lanka.
- **Email**: [pasindugunatilake144@gmail.com](mailto:pasindugunatilake144@gmail.com)
- **Portfolio**: [https://pasindugunathilake.github.io/Portfolio-PG](https://pasindugunathilake.github.io/Portfolio-PG)
- **Focus Areas**: AI/ML (Deep Learning & Computer Vision) Systems, System Architectures, RAG Systems, Full-Stack Development.

---

## License

This project is developed for educational, research, and open-source demonstration purposes under the **MIT License**.
