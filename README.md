# Smart Cooling Edge AI

An edge computer vision and deep learning system built for real-time classroom occupancy detection and automated HVAC energy saving.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.9-green?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/ONNX-Runtime-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv11-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MongoDB-Atlas-darkgreen?style=for-the-badge&logo=mongodb&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.0.2-lightgrey?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Enabled-blue?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-Educational-lightgrey?style=for-the-badge"/>
</p>

---

## 📌 Table of Contents

* [Overview](https://www.google.com/search?q=%23-overview)
* [System Architecture](https://www.google.com/search?q=%23-system-architecture)
* [Project Structure](https://www.google.com/search?q=%23-project-structure)
* [Tech Stack](https://www.google.com/search?q=%23-tech-stack)
* [Prerequisites](https://www.google.com/search?q=%23-prerequisites)
* [Installation](https://www.google.com/search?q=%23-installation)
* [Configuration](https://www.google.com/search?q=%23-configuration)
* [Running the Application](https://www.google.com/search?q=%23-running-the-application)
* [API Reference](https://www.google.com/search?q=%23-api-reference)
* [Troubleshooting](https://www.google.com/search?q=%23-troubleshooting)
* [License](https://www.google.com/search?q=%23-license)

---

## 🧠 Overview

The **Smart Cooling Edge AI** system is a lightweight, high-performance computer vision solution designed to optimize energy consumption in smart classrooms. Using a "Thin Edge" architecture, the edge device captures and streams live video feed to a central server where AI inference (via **YOLOv11 ONNX**) processes student occupancy counts. Based on dynamic occupancy brackets, the system automatically regulates HVAC parameters (temperatures, fan speeds, louver positions, and active AC units) and logs telemetry data securely to **MongoDB Atlas**.

---

### 🛡️ Core Principles

| Principle | Implementation |
| --- | --- |
| **Thin Edge Architecture** | Edge devices handle only capture and streaming; heavy AI inference runs on the central server. |
| **High Performance Inference** | Utilizes ONNX Runtime (`yolo11m.onnx`) for optimized CPU inference speed. |
| **Adaptive Image Enhancement** | Automatic CLAHE brightness adjustment for dark or shadowed classroom environments. |
| **Intelligent HVAC Control** | Matrix-based automation regulating 4 AC units according to live student count. |
| **Cloud Telemetry Logging** | Structured metadata and annotated frame paths logged directly to MongoDB Atlas. |
| **Modular Design** | Clean separation of components, utilities, database access, and core pipelines. |

---

## 🏗️ System Architecture

```
[Edge Node / Camera] 
       │ (HTTP MJPEG Stream)
       ▼
[Flask Backend Server] ──► [YOLO ONNX Predictor] ──► [HVAC Controller Matrix]
       │                                                         │
       ├─────────────────────────────────────────────────────────┤
       ▼                                                         ▼
[Live Stream Frontend]                                  [MongoDB Atlas Logger]

```

---

## 📁 Project Structure

```
Smart-cooling-edge_AI/
├── smart_cooling/                 # Core Package
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── model_predictor.py     # YOLO ONNX Inference & NMS logic
│   │   └── hvac_controller.py     # AC regulation matrix logic
│   ├── dbdata_access/
│   │   ├── __init__.py
│   │   └── data_logger.py         # MongoDB Atlas connection & logging
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── prediction_pipeline.py # Main orchestration loop (5-minute interval)
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py         # CLAHE preprocessing & frame annotations
│
├── models/                        # Pre-trained models directory
│   └── yolo11m.onnx
│
├── notebooks/                     # Jupyter notebooks for experiments
│   └── experiment.ipynb
│
├── app.py                         # Flask Backend & Dashboard Stream
├── edge_streamer.py               # Edge Node Script (Camera capture & stream)
├── setup.py                       # Package installation script
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container image definition
└── .env                           # Environment variables (Secrets)

```

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
| --- | --- | --- |
| **Edge Capture** | OpenCV (`cv2`) | Video frame capture and MJPEG streaming |
| **Inference Engine** | ONNX Runtime (`ort`) + Ultralytics YOLOv11 | Fast object detection and bounding box extraction |
| **Web Server** | Flask 3.0.2 + Flask-CORS | API routing and live dashboard streaming |
| **Database** | MongoDB Atlas (`pymongo` + `certifi`) | Cloud telemetry and historical log storage |
| **Image Processing** | NumPy + OpenCV | CLAHE enhancement, letterbox padding, and normalization |

---

## ✅ Prerequisites

Before setting up the project, ensure you have:

* **Python** 3.10 or higher installed.
* An active **MongoDB Atlas** cluster connection string.
* A connected webcam or video source for the edge node.

---

## 📦 Installation

### Step 1 — Clone and Navigate

Clone your repository and navigate into the project root directory.

### Step 2 — Create and Activate a Virtual Environment

```bash
python -m venv acvenv
# Windows:
acvenv\Scripts\activate
# macOS / Linux:
source acvenv/bin/activate

```

### Step 3 — Install Dependencies

Install the project in editable mode using `setup.py`:

```bash
pip install -e .

```

---

## 🔑 Configuration

Create a `.env` file in the root directory with your configuration parameters:

```env
MONGODB_URI="Enter your MongoDB Atlas connection string here"
DB_NAME="Edge_AI_Smart_Cooling"
COLLECTION_NAME="Edge_AI_Smart_Cooling_Logs"

```

---

## 🚀 Running the Application

### 1. Run the Edge Streamer (Edge Node)

On your edge device or simulation environment, start the video stream script:

```bash
python edge_streamer.py

```

### 2. Run the Central Flask Server

Start the backend server to handle inference, HVAC mapping, and database logging:

```bash
python app.py

```

---

## 🔌 API Reference

The Flask server exposes endpoints for managing live video streams and monitoring telemetry records.

* **`GET /video_feed`** — Streams the live processed MJPEG feed with bounding boxes and overlays.
* **`GET /status`** — Returns the latest HVAC control states and student counts in JSON format.

---

## 🔧 Troubleshooting

* **ModuleNotFoundError:** Ensure you ran `pip install -e .` so the `smart_cooling` package is properly registered in your environment.
* **MongoDB Timeout:** Verify your IP address is whitelisted in MongoDB Atlas and that `certifi` is installed.
* **Camera Not Opening:** Check your device index or replace the video source path in the configuration script.

---

## 📜 License

This project is developed for educational and operational research purposes.