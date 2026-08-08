import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from Smart_cooling_edge_AI.utils.image_utils import preprocess
from Smart_cooling_edge_AI.utils.configurations import read_yaml
from Smart_cooling_edge_AI.logging.logger import get_logger

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

# ONNX model params
MODEL_WEIGHTS_NAME = model_config.get('model_weights_name', 'yolo11m.pt')
IMG_SIZE = model_config.get('img_size', 640)
weights_path = onnx_path.parent / MODEL_WEIGHTS_NAME


# Model Download & ONNX export If Model does not exits
if not os.path.exists(onnx_path):
    logger.warning(f"ONNX model not found at {onnx_path}. Attempting automatic download and export...")
    try:
        from ultralytics import YOLO
        
        # Load pre-trained YOLO11m model (downloads 'yolo11m.pt' automatically if missing)
        logger.info(f"Loading pre-trained model weights from: {weights_path}")
        model = YOLO(str(weights_path))
        
        # Export to ONNX format
        logger.info(f"Exporting model to ONNX format with image size: {IMG_SIZE}")
        exported_file = model.export(format="onnx", imgsz=IMG_SIZE)
        
        # Move or rename the exported ONNX file to the target models directory
        if exported_file and os.path.exists(exported_file):
            import shutil
            if Path(exported_file).resolve() != onnx_path.resolve():
                shutil.move(exported_file, str(onnx_path))
                
        logger.info(f"ONNX model successfully generated and saved to: {onnx_path}")
    except Exception as e:
        logger.critical(f"Error during automatic model download/export: {str(e)}", exc_info=True)


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






## Detection function
def run_detection(frame):
    """
    Runs ONNX model inference and applies Non-Maximum Suppression (NMS).
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

        outputs = session.run([output_name], {input_name: input_tensor})
        preds = np.squeeze(outputs[0]).T

        boxes = []
        confidences = []
        class_ids = []

        for row in preds:
            classes_scores = row[4:]
            max_score = np.max(classes_scores)
            class_id = np.argmax(classes_scores)

            if class_id == 0 and max_score >= CONFIDENCE_THRESHOLD:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]

                x1 = int((cx - bw/2 - dw) / r)
                y1 = int((cy - bh/2 - dh) / r)
                x2 = int((cx + bw/2 - dw) / r)
                y2 = int((cy + bh/2 - dh) / r)

                w_box = x2 - x1
                h_box = y2 - y1

                boxes.append([x1, y1, w_box, h_box])
                confidences.append(float(max_score))
                class_ids.append(class_id)

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