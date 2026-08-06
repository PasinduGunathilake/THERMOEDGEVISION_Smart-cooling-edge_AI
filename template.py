import os 
from pathlib import Path

project_name = "Smart-cooling-edge_AI"

list_of_dirnfiles = [
    f"{project_name}/__init__.py",
    
    # Core Components (Inference & Smart Cooling Control)
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/model_predictor.py",
    f"{project_name}/components/hvac_controller.py",
    
    # Database Access (MongoDB)
    f"{project_name}/dbdata_access/__init__.py",
    f"{project_name}/dbdata_access/data_logger.py",
    
    # Pipeline (Main Loop)
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    
    # Utilities (Preprocessing & Annotations)
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/image_utils.py",
    
    # Root Level Files
    "models/yolo11m.onnx",
    "notebooks/experiment.ipynb",
    "app.py",                 # Flask Backend & Dashboard Stream
    "edge_streamer.py",       # Edge Node Script (Camera capture & stream)
    "templates/index.html",
    "requirements.txt",
    "Dockerfile",
    ".env",
    "setup.py",
    "demo.py"                 # Test Componets working correct or not
]

for filepath in list_of_dirnfiles:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        print(f"Creating dir: {filedir}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            print(f"Creating file: {filepath}")
    else:
        print(f"File already exists: {filename}")