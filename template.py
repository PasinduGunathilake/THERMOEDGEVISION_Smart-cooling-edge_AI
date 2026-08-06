import os 
from pathlib import Path

project_name = "Smart-cooling-edge_AI"

list_of_dirnfiles = [
    "notebooks/experiment.ipynb",
    "edge_node/__init__.py",
    "edge_node/inference.py",
    "backend/__init__.py",
    "backend/app.py",
    "frontend/templates/index.html",
    "models/modelOnnx",
    "edge_simulator.py",
    "requirements.txt",
    "setup.py",
    ".gitignore",
    ".env"
]

for filepath in list_of_dirnfiles:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        print(f"creating dir: {filedir}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            print(f"creating file: {filepath}")
    else:
        print(f"dile already exists: {filename}")
