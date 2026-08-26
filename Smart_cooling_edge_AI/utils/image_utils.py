import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from Smart_cooling_edge_AI.utils.configurations import read_yaml
from Smart_cooling_edge_AI.logging.logger import get_logger

# Initialize the logger
logger = get_logger("ImageUtils")

# Project root and Configurations
current_dir = Path(__file__).resolve().parent  # Smart_cooling_edge_AI/utils
project_root = current_dir.parent.parent       # Smart_cooling_edge_AI root

try:
    config = read_yaml(project_root / "params.yaml")
    logger.info("Configuration parameters loaded successfully in image_utils.")
except Exception as e:
    logger.critical(f"Failed to load params.yaml in image_utils: {str(e)}", exc_info=True)
    config = {}


def preprocess(frame, new_shape=None):
    """
    Complete preprocessing pipeline for YOLO models running via ONNX Runtime:
    1. BGR to RGB color conversion
    2. Optional CLAHE brightness enhancement for dark/shadowed images
    3. Aspect-ratio-preserving resizing with letterbox padding
    4. Normalization [0, 1]
    5. Transpose (HWC to CHW) and Batch dimension expansion
    """

    try:
        # Get img_size from params.yaml (default 640)
        if new_shape is None:
            img_dim = config['MODEL_CONFIG'].get('img_size', 640)
            new_shape = (img_dim, img_dim)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(frame_gray)

        if brightness < 80:
            logger.debug(f"Low brightness detected ({brightness:.2f}). Applying CLAHE enhancement.")
            lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            lab = cv2.merge((l, a, b))
            frame_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        height, width = frame_rgb.shape[:2]
        ratio = min(new_shape[0] / height, new_shape[1] / width)
        new_unpad = (int(round(width * ratio)), int(round(height * ratio)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]   
        dw /= 2   
        dh /= 2

        if (width, height) != new_unpad:
            im = cv2.resize(frame_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            im = frame_rgb

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))   
        
        img = im.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img, ratio, (dw, dh)

    except Exception as e:
        logger.error(f"Error occurred during frame preprocessing: {str(e)}", exc_info=True)
        return None, 1.0, (0, 0)





def draw_boxes(frame, boxes: list):
    """
    Draws YOLO bounding boxes and confidence labels onto a frame copy
    Returns the annotated frame (does not modify the original)
    """
    display_frame = frame.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4] if len(box) > 4 else 0.0

        # Bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Confidence label
        label        = f"student: {conf:.2f}"
        font_scale   = 0.6
        font_thick   = 2

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        label_bg_x2 = x1 + text_w + 5
        label_bg_y1 = y1 - text_h - 10

        # Label background rectangle
        cv2.rectangle(display_frame, (x1, label_bg_y1), (label_bg_x2, y1), (0, 0, 255), -1)

        # Label text
        cv2.putText(
            display_frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA
        )

    return display_frame


def save_annotated_frame(frame, detected_boxes, person_count, hvac_data):
    """
    Draws bounding boxes, overlays summary info, and saves (future model retraining) the frame locally
    """

    try:
        # Get frame save directory from params.yaml
        frame_save_dir = config['PIPELINE_CONFIG'].get('frame_save_dir', 'Frame_captures')
        capture_dir = project_root / frame_save_dir
        os.makedirs(capture_dir, exist_ok=True)

        frame_copy = draw_boxes(frame, detected_boxes)

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        temp_min = hvac_data.get('temp_min', hvac_data.get('target_temperature', 24.0))
        info_text = f"Time: {timestamp_str} | Count: {person_count} | Temp: {hvac_data['temp_min']}C"
        cv2.putText(frame_copy, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        filename = capture_dir / f"classroom_{file_timestamp_str}_count_{person_count}.jpg"
        cv2.imwrite(str(filename), frame_copy)

        logger.info(f"Captured frame saved locally: {filename}")

        return str(filename), timestamp_str

    except Exception as e:
        logger.error(f"Failed to save annotated frame: {str(e)}", exc_info=True)
        return "", datetime.now().strftime("%Y-%m-%d %H:%M:%S")