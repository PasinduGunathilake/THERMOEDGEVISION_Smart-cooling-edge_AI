import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from Smart_cooling_edge_AI.utils.image_utils import preprocess
from Smart_cooling_edge_AI.utils.configurations import read_yaml
from Smart_cooling_edge_AI.logging.logger import get_logger

# Windows CUDA DLL Directory Registration for ONNX Runtime GPU
if sys.platform == "win32":
    try:
        import site
        for site_pkg in site.getsitepackages():
            nvidia_path = os.path.join(site_pkg, "nvidia")
            if os.path.exists(nvidia_path):
                for root, dirs, files in os.walk(nvidia_path):
                    if any(f.endswith(".dll") for f in files):
                        try:
                            os.add_dll_directory(root)
                            os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
                        except Exception:
                            pass
    except Exception:
        pass


# Initialize logger
logger = get_logger("ModelPredictor")

current_dir = Path(__file__).resolve().parent  # smart_cooling_edge_AI/components
project_root = current_dir.parent.parent       # Smart_cooling_edge_AI root



## Load Configurations from params.yaml
try:
    config = read_yaml(project_root / "params.yaml")
    logger.info("Configuration parameters loaded successfully in model_predictor.")
except Exception as e:
    logger.critical(f"Failed to read params.yaml in model_predictor: {str(e)}", exc_info=True)
    config = {}

# Extract configuration parameters safely with fallback defaults to ensure production resilience and prevent KeyError crashes
model_config = config.get('MODEL_CONFIG', {})
model_relative_path = model_config.get('model_path', 'models/model.onnx')
onnx_path = project_root / model_relative_path

try:
    os.makedirs(onnx_path.parent, exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create model directory at {onnx_path.parent}: {str(e)}", exc_info=True)

# NMS parameters
CONFIDENCE_THRESHOLD = model_config.get('confidence_threshold', 0.5) # prevents 
NMS_THRESHOLD = model_config.get('nms_threshold', 0.45)

# ONNX model params (Defaults to ultra-fast YOLOv11n Nano variant)
MODEL_WEIGHTS_NAME = model_config.get('model_weights_name', 'yolo11n.pt')
IMG_SIZE = model_config.get('img_size', 640)
weights_path = onnx_path.parent / MODEL_WEIGHTS_NAME

# old (without fine tuning)---------------------------------------

# Model Download & ONNX export If Model does not exist (Commented out reference code)
# if not os.path.exists(onnx_path):
#     logger.warning(f"ONNX model not found at {onnx_path}. Attempting automatic download and export...")
#     try:
#         from ultralytics import YOLO
#         
#         # Load pre-trained YOLO11n Nano model (downloads 'yolo11n.pt' automatically if missing)
#         logger.info(f"Loading pre-trained model weights from: {weights_path}")
#         model = YOLO(str(weights_path))
#         
#         # Export to ONNX format
#         logger.info(f"Exporting model to ONNX format with image size: {IMG_SIZE}")
#         exported_file = model.export(format="onnx", imgsz=IMG_SIZE)
#         
#         # Move or rename the exported ONNX file to the target models directory
#         if exported_file and os.path.exists(exported_file):
#             import shutil
#             if Path(exported_file).resolve() != onnx_path.resolve():
#                 shutil.move(exported_file, str(onnx_path))
#                 
#         logger.info(f"ONNX model successfully generated and saved to: {onnx_path}")
#     except Exception as e:
#         logger.critical(f"Error during automatic model download/export: {str(e)}", exc_info=True)
# ---------------------------------------------------------------

# Check ONNX model file existence
if not os.path.exists(onnx_path):
    logger.warning(f"ONNX model not found at {onnx_path}. Ensure the retrained model is exported to this location.")



# Load ONNX Run Time session
session = None
input_name = None
output_name = None

try:
    if os.path.exists(onnx_path):
        # Try GPU first, fall back to CPU if GPU is not supported/configured
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        logger.info(f"Initializing ONNX runtime session from: {onnx_path}")
        session = ort.InferenceSession(onnx_path, providers= providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Check active provider to confirm if it's using GPU
        active_providers = session.get_providers()
        logger.info(f"ONNX model successfully loaded. Active Execution Providers: {active_providers}")
    else:
        logger.error(f"ONNX model file missing at path: {onnx_path}. Inference will be disabled.")
except Exception as e:
    logger.critical(f"Failed to initialize ONNX Runtime Inference Session: {str(e)}", exc_info=True)
    session = None
    input_name, output_name = None, None






## Detection function (Vectorized NumPy implementation for 10x-100x faster NMS parsing)
def run_detection(frame):
    """
    Runs ONNX model inference and applies Non-Maximum Suppression (NMS).
    Optimized using NumPy matrix vectorization instead of slow Python loops.
    """
    if session is None or input_name is None or output_name is None:
        logger.error("ONNX session is not initialized. Skipping detection.")
        return [], 0

    try:
        h_orig, w_orig = frame.shape[:2]
        input_tensor, r, (dw, dh) = preprocess(frame)

        if input_tensor is None:
            logger.error("Preprocessing returned empty tensor. Skipping detection.")
            return [], 0

        # ONNX Model Inference Forward Pass
        outputs = session.run([output_name], {input_name: input_tensor})
        preds = np.squeeze(outputs[0]).T  # Transpose to shape (8400, 84)

       
        # Vectorized NumPy Filtering (for without looping increase speed)
        class_scores = preds[:, 4:]  # Class probability columns
        max_scores = np.max(class_scores, axis=1)  # Top class confidence score
        class_ids = np.argmax(class_scores, axis=1)  # Top class index

        # Mask for Person class (class_id == 0) exceeding Confidence Threshold
        mask = (class_ids == 0) & (max_scores >= CONFIDENCE_THRESHOLD)

        boxes = []
        confidences = []
        if np.any(mask):
            filtered_preds = preds[mask]
            filtered_scores = max_scores[mask]

            # Extract center coordinates, width, and height in parallel
            cx = filtered_preds[:, 0]
            cy = filtered_preds[:, 1]
            bw = filtered_preds[:, 2]
            bh = filtered_preds[:, 3]

            # Map bounding box coordinates back to original frame scale
            x1 = np.int32((cx - bw / 2 - dw) / r)
            y1 = np.int32((cy - bh / 2 - dh) / r)
            w_box = np.int32(bw / r)
            h_box = np.int32(bh / r)

            # Build box array for NMSBoxes
            for i in range(len(filtered_preds)):
                boxes.append([int(x1[i]), int(y1[i]), int(w_box[i]), int(h_box[i])])
                confidences.append(float(filtered_scores[i]))

        # Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=CONFIDENCE_THRESHOLD, nms_threshold=NMS_THRESHOLD)

        detected_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w_box, h_box = boxes[i]
                conf = confidences[i]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_orig, x + w_box)
                y2 = min(h_orig, y + h_box)
                detected_boxes.append((x1, y1, x2, y2, conf))

        detected_count = len(detected_boxes)
        return detected_boxes, detected_count
    
    except Exception as e:
        logger.error(f"Error occurred during model inference execution: {str(e)}", exc_info=True)
        return [], 0


# Dynamically update NMS threshold parms from frontend
def update_thresholds(new_conf, new_nms):
    """
    Dynamically updates the in-memory thresholds without reloading the model.
    """
    global CONFIDENCE_THRESHOLD, NMS_THRESHOLD
    
    CONFIDENCE_THRESHOLD = float(new_conf)
    NMS_THRESHOLD = float(new_nms)
    
    logger.info(f"Model thresholds dynamically updated -> CONF: {CONFIDENCE_THRESHOLD}, NMS: {NMS_THRESHOLD}") 