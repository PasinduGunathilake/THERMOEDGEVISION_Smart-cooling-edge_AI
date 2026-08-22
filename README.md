<div align="center">

# THERMOEDGEVISION - Smart Cooling Edge AI

**Intelligent Real-Time Classroom Occupancy Detection & Automated AC Energy Optimization at the Edge**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ultralytics YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-purple?style=for-the-badge)](https://ultralytics.com/)
[![ONNX Runtime GPU](https://img.shields.io/badge/ONNX_Runtime-CUDA_GPU-orange?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA_CUDA-11.8%20%2F%2012.0-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.2-lightgrey?style=for-the-badge&logo=flask&logoColor=black)](https://flask.palletsprojects.com/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-cyan?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Atlas_Cloud-darkgreen?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-9B51E0?style=for-the-badge&logo=roboflow&logoColor=white)](https://roboflow.com/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## Project Overview

**Smart Cooling Edge AI** is a lightweight, high-performance edge computing solution designed to optimize AC energy consumption across smart university classrooms and auditoriums. 

Instead of running central AC units at full blast in empty or half-filled rooms, the system captures real-time camera feeds, performs rapid student detection using **YOLOv11 ONNX Runtime GPU Acceleration**, and dynamically regulates AC units (temperature, fan speed, louver position) based on live student count. All telemetry events and detection snapshots are securely logged to **MongoDB Atlas Cloud** to establish a **data flywheel** enabling **long-term analytics, future ML-driven predictive control (such as anticipating occupancy trends), and environmental footprint auditing.**

---

## Key Features

- **Asynchronous High-Performance Inference:** Dedicated threading for YOLOv11 ONNX inference prevents blocking the video feed, resulting in ultra-smooth real-time display (30+ FPS) while maintaining robust detection.
- **Adaptive CLAHE Enhancement:** Automatic contrast limited adaptive histogram equalization for dark, shadowed, or poorly lit classroom environments.
- **Intelligent AC Automation:** Matrix-based rule engine dynamically regulates up to 4 AC units according to live student count.
- **Persistent Video Stream Playback:** Video stream playback continues seamlessly from where it left off across web UI page navigations.
- **Cloud Telemetry Logging:** Push structured occupancy and climate control data directly to MongoDB Atlas Cloud over secure TLS SSL.

---

## System Architecture & Backend Data Flow

This diagram illustrates how raw video inputs move through the **Flask Server**, **AI Inference Worker**, **AC Control Engine**, and **MongoDB Cloud**:

<div align="center">

[![System Architecture](./resources/Diagram/THERMOEDGEVISION%20smart%20coolin%20ai.jpg)](./resources/Diagram/THERMOEDGEVISION%20smart%20coolin%20ai.jpg)

> *Click on the diagram image above to open and zoom into the full high-resolution architecture visual.*

</div>

### Architecture Component Breakdown:

1. **Input Layer**: Ingests camera streams or uploaded video files directly into the Flask backend (`app.py`).
2. **Display Thread (`prediction_pipeline.py`)**: Handles smooth video rendering (1280x720), overlays bounding boxes, and streams HTTP MJPEG to the web browser at **30+ FPS**.
3. **Inference Thread (`inference_thread.py`)**: Runs in the background without causing video lag:
   - **Adaptive CLAHE Brightness Enhancer**: Automatically brightens dark classroom frames (`brightness < 80`).
   - **YOLOv11 ONNX GPU Engine**: Detects students under CUDA GPU acceleration (`~4.2 ms`).
   - **NMS Filter**: Rapidly filters detection coordinates using vectorized NumPy math via Non-Maximum Suppression.
4. **AC Control Engine (`cooling_controller.py`)**: Evaluates student count and automatically adjusts active AC units (1-4), target temperatures (17°C - 25°C), fan speeds, and louvers.
5. **Cloud Logger (`data_logger.py`)**: Asynchronously pushes occupancy telemetry and image snapshots to **MongoDB Atlas Cloud** for future analytics or ML model training for contorl ac system without predefined ruls.

---

## MLOps Practices & Production Standards

This project strictly adheres to core **MLOps (Machine Learning Operations)** standards to ensure scalability, maintainability, and high performance in production:

- **Model Optimization:** Transitioned from raw PyTorch models to **ONNX Runtime Engine**, providing faster, hardware-agnostic inference on edge CPUs/GPUs with automatic CPU fallback if GPU hardware is absent, featuring a compact **10.2 MB** binary footprint.
- **Configuration Management:** Centralized all detection thresholds (NMS, SCORE-confidence) and AC automation rules in `params.yaml`. This avoids hard coded values and enables reproducible model tracking.
- **Data Flywheel / Experiment Tracking:** Continuous logging of telemetry data and annotated images to **MongoDB Atlas**. This establishes a robust data pipeline for future model retraining, drift analysis, and dashboarding.
- **Decoupled Architecture:** Utilizes a producer-consumer threading model to completely separate IO-bound stream display from compute-bound YOLO inference, maximizing real-time FPS.
- **Secure Secrets Management:** Use of `.env` files ensures no database credentials or API keys are committed to version control.
- **Standardized Python Packaging:** Uses `setup.py` and `requirements.txt` to register the local package (`Smart_cooling_edge_AI`), ensuring clean imports and modularity.

---

## Model Training, Fine-Tuning & Research Experiments

Model fine-tuning, benchmarking, and quantization workflows are conducted in the `notebooks/` directory:

### 1. YOLO11 Retraining & ONNX Export Pipeline (`notebooks/train_yolo11.ipynb`)
- **Dataset Ingestion:** Downloads and extracts annotated classroom student occupancy datasets via Roboflow API.
- **Model Fine-Tuning :** Fine-tunes base YOLO11 Nano (`yolo11n.pt`) weights specifically for single-class (`person`) classroom detection across 100-300 epochs using AdamW optimizer.
- **Hardware Acceleration:** Supports GPU execution (`device=0` with CUDA) for high-speed training and inference, with automatic graceful fallback to CPU if CUDA GPU hardware is absent. Includes CPU thread throttling (`torch.set_num_threads`) to prevent thermal throttling on CPU-only setups.
- **Automated ONNX Export:** Automatically exports the retrained PyTorch weights to optimized 640x640 ONNX format (`models/yolo11n.onnx`) and deploys them to the production models path.

### 2. Research & Quantization Experiments (`notebooks/experiment.ipynb`)
- **Model Quantization:** Converts PyTorch models to ONNX runtime format, compressing binary footprint to **10.2 MB**.
- **Execution Provider Benchmarking:** Compares `CUDAExecutionProvider` vs `CPUExecutionProvider` (achieving 3x-5x GPU speedup).
- **CLAHE Enhancement:** Benchmarks LAB color space Contrast Limited Adaptive Histogram Equalization for low-light classroom vision.
- **Vectorized NMS Post-Processing:** Implements NumPy matrix filtering (`preds[:, 4:]`) over 8,400 raw detection anchors for 10x-100x faster execution.

---

## Web UI Pages & Dashboard

Built with a sleek scifi glassmorphic design system using Tailwind CSS and monospaced typography:

- **Landing Page (`/`)**: 10-second dynamic loading screen.
- **Enterprise Dashboard (`/dashboard`)**: Real-time video player, 4 AC Unit control cards with rotating SVG turbine discs, target temp, and edge hardware widget.
- **Edge Nodes Matrix (`/nodes`)**: Live CPU/GPU utilization, RAM/VRAM, ONNX execution provider, model size, inference speed, host IP, and active subsystem cards.
- **Analytics & DB Logs (`/analytics`)**: Live streamed execution logs terminal and MongoDB Atlas document cards.
- **Settings (`/settings`)**: Dynamic detection threshold sliders synced with `params.yaml`.

---

## AC Automation Rule

AC units and climate settings adjust dynamically based on student count rules defined in `params.yaml`:

| Occupancy Bracket | Student Count | Active ACs | Mode | Target Temp | Fan Speed | Louver |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Empty** | `0` | `1 Unit` | `Baseline` | `25°C - 26°C` | `Low` | `Down` |
| **Low** | `1 - 20` | `2 Units` | `Alternate` | `24°C - 25°C` | `Low` | `Down` |
| **Medium** | `21 - 60` | `4 Units` | `Normal` | `22°C - 23°C` | `Medium` | `Swing` |
| **High** | `61 - 110` | `4 Units` | `Active` | `20°C - 21°C` | `High` | `Horizontal` |
| **Full / Turbo** | `111 - 150+` | `4 Units` | `Turbo` | `17°C - 19°C` | `Max` | `All Swing` |

---

## Flask REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Landing loading screen (`loading.html`) |
| `/dashboard` | `GET` | Main control dashboard (`dashboard.html`) |
| `/nodes` | `GET` | Edge hardware telemetry page (`nodes.html`) |
| `/analytics` | `GET` | Analytics & MongoDB logs page (`analytics.html`) |
| `/settings` | `GET` | System configuration page (`settings.html`) |
| `/video_feed` | `GET` | MJPEG annotated live video stream |
| `/api/upload_video` | `POST` | Upload video file and switch stream source |
| `/api/stop_stream` | `POST` | Stop stream and clean temporary video file |
| `/api/status` | `GET` | Real-time hardware telemetry API (CPU, GPU, RAM, ONNX, IP) |
| `/api/analytics_logs` | `GET` | Fetch latest 15 telemetry logs from MongoDB Cloud |
| `/api/system_logs` | `GET` | Stream system execution log lines |
| `/api/update_thresholds` | `POST` | Dynamically update NMS & confidence detection sliders in `params.yaml` |

---

## Project Directory Structure

```text
Smart-cooling-edge_AI/
├── Smart_cooling_edge_AI/                # Core Application Package
│   ├── components/                       # Inference & Cooling Logic
│   │   ├── model_predictor.py            # ONNX Inference Session & Vectorized NMS
│   │   └── cooling_controller.py         # AC Automation Rule Engine
│   ├── dbdata_access/                    # MongoDB Atlas Cloud Logging
│   ├── logging/                          # System Execution Logger
│   ├── pipeline/                         # Multithreaded Pipeline
│   │   ├── inference_thread.py           # Asynchronous Worker Thread
│   │   └── prediction_pipeline.py        # Main Thread Display Pipeline
│   └── utils/                            # Image Preprocessing & Config Parsers
│       ├── image_utils.py                # CLAHE Contrast Enhancement & Letterboxing
│       └── configurations.py             # YAML Parameter Reader
├── models/                               # Exported Production ONNX Models (`yolo11n.onnx` - 10.2 MB)
├── notebooks/                            # Model Training & Experiments (`train_yolo11.ipynb`, `experiment.ipynb`)
├── raw_data/                             # Raw Training Data & Frame Ingestion
│   ├── videos/                           # Input Raw Classroom Videos
│   └── raw_frames/                       # Extracted Frames for Roboflow Annotation
├── templates/                            # Web UI Templates (`dashboard.html`, etc.)
├── static/                               # Web UI Static Assets (CSS, JS, Icons)
├── resources/                            # Architecture Visual Diagrams
│   └── Diagram/                          # System Architecture Visual Diagram
│       └── THERMOEDGEVISION smart coolin ai.jpg
├── Frame_captures/                       # Saved Retraining (future) Frame Snapshots
├── temp_uploads/                         # Temporary Video Upload Storage
├── logs/                                 # System Execution Logs
├── app.py                                # Main Flask Backend Web Server
├── demo.py                               # Standalone Prediction Pipeline Demo Runner
├── extract_frames.py                     # Automatic Video Frame Extractor Utility(For YOLO retraining)
├── template.py                           # Automated Project Structure Generator
├── setup.py                              # Package Installation Script
├── requirements.txt                      # Project Dependencies
├── params.yaml                           # System Configuration Parameters
├── Dockerfile                            # Docker Containerization Specification
├── .dockerignore                         # Docker Build Ignore Rules
├── .gitignore                            # Git Version Control Ignore Rules
├── .env                                  # Environment Variables & Secret Credentials
├── LICENSE                               # License
└── README.md                             # Comprehensive Project Documentation
```

---

## Installation & Setup

### 1. Prerequisites
Before setting up the project, ensure you have:
* **Python 3.10+**
* **NVIDIA GPU drivers & CUDA toolkit** (Optional - ONNX Runtime automatically falls back to CPU execution if GPU hardware is not present/ Not Essential)
* An active **MongoDB Atlas** cluster connection string.

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/PasinduGunathilake/Smart-cooling-edge_AI.git
cd Smart-cooling-edge_AI

# Windows
python -m venv < virtual_environment_name >
< virtual_environment_name >\Scripts\activate

# macOS / Linux
python3 -m venv < virtual_environment_name >
source < virtual_environment_name >/bin/activate
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

---

## Running the Application

### Option A: Production Web App Server (Recommended)
Start the central Flask backend server and open the web UI dashboard:
```bash
python app.py
```
Open your browser at: `http://127.0.0.1:5000`

### Option B: Standalone Pipeline OpenCV Test
To run the full prediction pipeline locally with a direct OpenCV window display (without web UI):
```bash
python -c "from Smart_cooling_edge_AI.pipeline.prediction_pipeline import PredictionPipeline; PredictionPipeline().run_pipeline()"
```

### Option C: Component & Pipeline Demo Runner
To execute the automated demo test script:
```bash
python demo.py
```

### Option D: Docker Container Deployment
The project includes a production-ready `Dockerfile` based on `python:3.10-slim` pre-configured with OpenCV native C-libraries (`ffmpeg`, `libgl1`, `libglib2.0-0`).

1. **Build Docker Image:**
   ```bash
   docker build -t smart-cooling-edge-ai .
   ```

2. **Run Docker Container:**
   Pass database environment variables via `.env` and map Flask port `5000`:
   ```bash
   docker run -d -p 5000:5000 --env-file .env --name smart-cooling-app smart-cooling-edge-ai
   ```

3. **Verify Deployment & Container Logs:**
   Open `http://localhost:5000` in your browser or stream active container execution logs:
   ```bash
   docker logs -f smart-cooling-app
   ```

---

## Model Fine-Tuning & ReTraining Guide

Follow these steps to annotate images, re-train, or fine-tune the YOLOv11 model on a custom classroom dataset:

### 1. Dataset Annotation & Preparation
If you wish to collect and annotate a new custom dataset for a specific classroom environment:
1. **Raw Video Placement:** Place raw classroom MP4/AVI/....  videos into `raw_data/videos/`.
2. **Automatic Frame Extraction:** Run `extract_frames.py` to sample images into `raw_data/raw_frames/` at 300-frame(You can change that needed) sampling intervals:
   ```bash
   python extract_frames.py
   ```
3. **Bounding Box Annotation:** Upload extracted frames from `raw_data/raw_frames/` to [Roboflow](https://roboflow.com/). Draw tight bounding box annotations around each student/person and label the class as `person`.
4. **Dataset Export:** Export the dataset in **YOLOv11** format. Copy the dataset zip / `data.yaml` location for use in the training pipeline.

### 2. Execution Environment (Local GPU vs Google Colab)
- **Local NVIDIA GPU (Recommended):** Run locally with CUDA support (`device=0`) in a PyTorch GPU environment (e.g., `torchenv`).
- **Google Colab (Free T4 GPU Alternative):** If your local laptop lacks an NVIDIA GPU, upload `notebooks/train_yolo11.ipynb` to [Google Colab](https://colab.research.google.com/), select **Runtime -> Change runtime type -> T4 GPU**, and execute the training pipeline. Download the exported `yolo11n.onnx` file from Colab and place it at `models/yolo11n.onnx` in your project folder.

---

### Step-by-Step Training Steps:

### 1. Open the Training Notebook
Launch Jupyter Notebook or VS Code and open `notebooks/train_yolo11.ipynb`:
```bash
jupyter notebook notebooks/train_yolo11.ipynb
```

### 2. Select Python Kernel
In VS Code or Jupyter, select a Python environment with GPU/CUDA support (e.g., `Python`).

### 3. Step 1: Download Roboflow Dataset
Provide your Roboflow API key and download the target version:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("your-workspace").project("smart_cooling_edge_ai")
dataset = project.version(1).download("yolov11")
```

### 4. Step 2: Train Model
Train YOLOv11 Nano with CUDA acceleration (`device=0`) across 100-300 epochs:
```python
import torch
from ultralytics import YOLO

# Detect GPU or CPU
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Training device: {device}")

model = YOLO("yolo11n.pt")

results = model.train(
    data="./smart_cooling_edge_ai-1/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    workers=4,
    device=device,
    name="yolo11n_smart_cooling_retrained"
)
```

> **Note for CPU-only training:** To prevent CPU overheating or thermal reboots on laptops without NVIDIA GPUs, set `torch.set_num_threads(4)` and reduce `batch=8`.

### 5. Step 4: Export & Deploy ONNX Model
Export the fine-tuned PyTorch checkpoint (`best.pt`) to ONNX format and copy it to production:
```python
import os
import shutil
from ultralytics import YOLO

best_weights_path = "runs/detect/yolo11n_smart_cooling_retrained/weights/best.pt"
model = YOLO(best_weights_path)

onnx_file_path = model.export(format="onnx", imgsz=640, simplify=True)
target_onnx_path = "../models/yolo11n.onnx"

os.makedirs(os.path.dirname(target_onnx_path), exist_ok=True)
shutil.copy(onnx_file_path, target_onnx_path)
print(f"[✔] Successfully deployed production ONNX model to: {target_onnx_path}")
```

Once exported, restart the Flask Web Backend (`python app.py`) to immediately begin real-time inference using the newly retrained ONNX model.

---

## Troubleshooting

- **Double Inference / Video Lag:** The system utilizes a threaded architecture. If lag persists, verify GPU execution by ensuring `onnxruntime-gpu` is installed and check GPU usage via `/nodes`.
- **MongoDB Connection Timeout:** Verify your current IP address is whitelisted in MongoDB Atlas Network Access and ensure `certifi` is installed (`pip install certifi`).
- **Camera / Video Stream Not Opening:** Check `video_source` in `params.yaml` under `PIPELINE_CONFIG`. Update `0` to your specific camera device index, RTSP stream URL, or video file path.

---

## Contributing

Contributions, issues, and feature requests are welcome! Key areas for contribution include:
- **ML Training Pipeline:** Build automated data extraction pipelines using MongoDB telemetry logs to prepare custom training datasets.
- **Model Retraining & Fine-Tuning:** Fine-tune default YOLO models using domain-specific classroom capture datasets for higher detection recall.
- **System Optimization:** Optimize multi-threading performance, edge stream latency, and ONNX runtime execution speeds.

#### Workflow for Contributing:
1. Fork the Project
2. Create your Branch (`git checkout -b feature/BranchName`)
3. Commit your Changes (`git commit -m 'Add some bla.. bla.. bla..'`)
4. Push to the Branch (`git push origin feature/BranchName`)
5. Open a Pull Request

---

## Author

### Pasindu Gunathilake
- **Education**: BSc. Computer Science Undergraduate, University of Jaffna, Sri Lanka.
- **Email**: [pasindugunatilake144@gmail.com](mailto:pasindugunatilake144@gmail.com)
- **Portfolio**: [https://pasindugunathilake.github.io/Portfolio-PG](https://pasindugunathilake.github.io/Portfolio-PG)
- **Focus Areas**:  AI/ML (with Deep Learning & Computer Vision) Systems, Edge AI Architectures, RAG Systems, Full-Stack Development.

---

## License

This project is developed for educational, research, and open-source demonstration purposes under the MIT License.
