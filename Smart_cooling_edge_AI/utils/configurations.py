import yaml
from pathlib import Path
from Smart_cooling_edge_AI.logging.logger import get_logger

# initialize logger
logger = get_logger("Configurations")

def read_yaml(path_to_yaml: Path) -> dict:
    """Reads a YAML configuration file and returns its content as a dictionary."""
    try:
        logger.info(f"Attempting to read YAML file from: {path_to_yaml}")

        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Successfully read and parsed YAML file: {path_to_yaml}")
            return content
            return content
    except Exception as e:
        logger.error(f"Failed to read YAML configuration file at {path_to_yaml}: {str(e)}", exc_info=True)
        raise e