import os 
from pathlib import Path

project_name = "Smart_cooling_edge_AI"

list_of_dirnfiles = [
    f"{project_name}/__init__.py",
    
    # Core Components (Inference & Smart Cooling Control)
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/model_predictor.py",
    f"{project_name}/components/cooling_controller.py",
    
    # Database Access (MongoDB)
    f"{project_name}/dbdata_access/__init__.py",
    f"{project_name}/dbdata_access/data_logger.py",
    
    # Pipeline (Main Loop)
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    
    # Utilities (Preprocessing & Annotations / Confugurations (YAML parameters reading) utility/ aviod hardcoding)
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/image_utils.py",
    f"{project_name}/utils/configurations.py",

    # Logging (System Logging, Trace , Security Levels Track)
    f"{project_name}/logging/__init__.py",
    f"{project_name}/logging/logger.py",
    
    # Root Level Files
    "models/model.onnx",
    "notebooks/experiment.ipynb",
    "app.py",                 # Flask Backend & Dashboard Stream
    "edge_streamer.py",       # Edge Node Script (Camera capture & stream)
    "templates/index.html",
    "params.yaml",            # Configuration (YAML params) management
    "requirements.txt",
    "Dockerfile",
    ".env",
    "setup.py",
    "demo.py"                 # Test Componets working correct or not
    #"resources",              # Sample images/videos (add manually)
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