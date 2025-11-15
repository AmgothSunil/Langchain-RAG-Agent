import os
import sys
import yaml
import logging

from app.core.exception import AppException

logger = logging.getLogger(__name__)

def load_params(params_path: str) -> dict:
    """
    Load application configuration parameters from a YAML file.

    Args:
        params_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed YAML content as a Python dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        Exception: For all other unexpected errors.
    """
    
    if not os.path.exists(params_path):
        logger.error("Configuration file not found: %s", params_path)
        raise FileNotFoundError(f"Configuration file not found: {params_path}")

    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        if not params:
            logger.warning("YAML file %s is empty.", params_path)
        else:
            logger.debug("Parameters retrieved successfully from %s", params_path)
        return params
    
    except yaml.YAMLError as e:
        logger.error("YAML parsing error in file %s: %s", params_path, e)
        raise

    except Exception as e:
        logger.error("Unexpected error while reading YAML file %s: %s", params_path, e)
        raise AppException(e, sys)