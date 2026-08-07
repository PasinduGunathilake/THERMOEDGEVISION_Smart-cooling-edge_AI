import os
import cv2
import numpy as np
from datetime import datetime

def preprocess(frame, new_shape=(640, 640)):
    """
    Complete preprocessing pipeline for YOLO models running via ONNX Runtime:
    1. BGR to RGB color conversion
    2. Optional CLAHE brightness enhancement for dark/shadowed images
    3. Aspect-ratio-preserving resizing with letterbox padding
    4. Normalization [0, 1]
    5. Transpose (HWC to CHW) and Batch dimension expansion
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(frame_gray)

    if brightness < 80:
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




def save_annotated_frame(frame, detected_boxes, person_count, hvac_data):
    """
    Draws bounding boxes, overlays summary info, and saves the frame locally.
    """
    capture_dir = "Frame_captures"
    os.makedirs(capture_dir, exist_ok=True)

    frame_copy = frame.copy()

    for box in detected_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4] if len(box) > 4 else 0.0
        
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"conf: {conf:.2f}"
        
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_copy, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), (0, 0, 255), -1)
        cv2.putText(frame_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    info_text = f"Time: {timestamp_str} | Count: {person_count} | Temp: {hvac_data['temp_min']}C"
    cv2.putText(frame_copy, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    filename = f"{capture_dir}/classroom_{file_timestamp_str}_count_{person_count}.jpg"
    cv2.imwrite(filename, frame_copy)
    print(f"Captured frame saved locally: {filename}")
    
    return filename, timestamp_str