import logging
import sys
from pathlib import Path
from datetime import datetime

def get_logger(name: str):
    """
    Centralized logger that automatically generates a unique log file name 
    using the current timestamp for every system run.
    """
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Console Handler - for Display in terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File Handler with Timestamp based filename
    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"log_{timestamp_str}.log"
    
    full_log_path = log_dir / log_filename
    main_log_path = log_dir / "system_execution.log"
    
    file_handler = logging.FileHandler(full_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    main_file_handler = logging.FileHandler(main_log_path, encoding="utf-8")
    main_file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    main_file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(main_file_handler)
    
    return logger