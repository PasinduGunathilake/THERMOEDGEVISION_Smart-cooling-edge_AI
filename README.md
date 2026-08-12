<div align="center">

# THERMOEDGEVISION - Smart Cooling Edge AI

**Intelligent Real-Time Classroom Occupancy Detection & Automated AC Energy Optimization at the Edge**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ultralytics YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-purple?style=for-the-badge)](https://ultralytics.com/)
[![ONNX Runtime GPU](https://img.shields.io/badge/ONNX_Runtime-CUDA_GPU-orange?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.2-lightgrey?style=for-the-badge&logo=flask&logoColor=black)](https://flask.palletsprojects.com/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-cyan?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Atlas_Cloud-darkgreen?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
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
5. **Cloud Logger (`data_logger.py`)**: Asynchronously pushes occupancy telemetry and image snapshots to **MongoDB Atlas Cloud**.

---

## MLOps Practices & Production Standards

This project strictly adheres to core **MLOps (Machine Learning Operations)** standards to ensure scalability, maintainability, and high performance in production:

- **Model Optimization:** Transitioned from raw PyTorch models to **ONNX Runtime Engine**, providing faster, hardware agnostic inference on edge CPU/GPUs with a compact **10.2 MB** binary footprint.
- **Configuration Management:** Centralized all detection thresholds (NMS, SCORE-confidence) and AC automation rules in `params.yaml`. This avoids hard coded values and enables reproducible model tracking.
- **Data Flywheel / Experiment Tracking:** Continuous logging of telemetry data and annotated images to **MongoDB Atlas**. This establishes a robust data pipeline for future model retraining, drift analysis, and dashboarding.
- **Decoupled Architecture:** Utilizes a producer-consumer threading model to completely separate IO-bound stream display from compute-bound YOLO inference, maximizing real-time FPS.
- **Secure Secrets Management:** Use of `.env` files ensures no database credentials or API keys are committed to version control.
- **Standardized Python Packaging:** Uses `setup.py` and `requirements.txt` to register the local package (`Smart_cooling_edge_AI`), ensuring clean imports and modularity.

---

## Jupyter Notebook Research (`notebooks/experiment.ipynb`)

Initial prototyping and experiments were conducted in `notebooks/experiment.ipynb`:

- **Model Quantization:** Converted PyTorch (`yolo11n.pt`) to ONNX (`yolo11n.onnx`), shrinking weights to **10.2 MB**.
- **Execution Provider Testing:** Benchmarked `CUDAExecutionProvider` vs `CPUExecutionProvider` (3x-5x speedup under GPU).
- **CLAHE Enhancement:** Tested LAB color space contrast enhancement for low-light classroom frames.
- **Vectorized NMS Post-Processing:** Applied NumPy vectorized matrix filtering (`preds[:, 4:]`) over 8,400 raw detection anchors for 10x-100x faster execution.

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
| **Low** | `0 - 20` | `2 Units` | `Alternate` | `24°C - 25°C` | `Low` | `Down` |
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

---

## Project Directory Structure

```text
Smart-cooling-edge_AI/
├── Smart_cooling_edge_AI/                # Core Application Package
│   ├── components/                       # Inference & Cooling logic
│   │   ├── model_predictor.py            # ONNX Session & Detection
│   │   ├── cooling_controller.py         # AC Automation Rules
│   │   └── yolo11n.onnx                  # Exported ONNX Model (10.2 MB)
│   ├── dbdata_access/                    # MongoDB Atlas logging
│   ├── logging/                          # System logger config
│   ├── pipeline/                         # Multithreaded Pipeline
│   │   ├── inference_thread.py           # Background Thread
│   │   └── prediction_pipeline.py        # Main Thread Display Pipeline
│   └── utils/                            # Image Preprocessing & Config Parsers
├── models/                               # Pretrained models (.onnx)
├── notebooks/                            # Jupyter Notebook Experiments (`experiment.ipynb`)
├── templates/                            # HTML UI Templates (`dashboard.html`, etc.)
├── resources/                            # Assets
│   └── Diagram/                          # System Architecture Visual Diagram
│       └── THERMOEDGEVISION smart coolin ai.jpg
├── Frame_captures/                       # Saved Retraining Frame Snapshots
├── temp_uploads/                         # Temporary Video Storage
├── logs/                                 # System Execution Logs
├── app.py                                # Main Flask Backend Web Server
├── setup.py                              # Package Installation Script
├── requirements.txt                      # Dependencies
├── params.yaml                           # System Configs
├── .env                                  # Environment Variables 
└── README.md                             # Documentation
```

---

## Installation & Setup

### 1. Prerequisites
Before setting up the project, ensure you have:
* **Python 3.10+**
* **NVIDIA GPU drivers & CUDA toolkit** (Optional but recommended for `onnxruntime-gpu`)
* An active **MongoDB Atlas** cluster connection string.

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/PasinduGunathilake/Smart-cooling-edge_AI.git
cd Smart-cooling-edge_AI

# Windows
python -m venv acvenv
acvenv\Scripts\activate

# macOS / Linux
python3 -m venv acvenv
source acvenv/bin/activate
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
COLLECTION_NAME="telemetry_logs"
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
- **Focus Areas**: Full-Stack Development, AI/ML (with Deep Learning) Systems, RAG Systems and Architectures.

---

## License

This project is developed for educational, research, and open-source demonstration purposes under the MIT License.
