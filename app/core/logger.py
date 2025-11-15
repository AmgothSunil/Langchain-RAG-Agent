import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of rotated logs to keep

def setup_logger(
    name: str = None,
    log_file_name: str = None,
) -> logging.Logger:
    """
    Creates and returns a logger instance with console + file handlers.
    Rotates logs automatically when file grows large.

    Args:
        name (str): Logger name (usually module name).
        log_dir_path (str): Directory for log files.
        log_file_name (str): Log file name.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """

    if not log_file_name:
        log_file_name = f"{__name__.replace('.', '_')}.log"

    log_dir_path = os.path.join(os.getcwd(), LOG_DIR) 
    os.makedirs(log_dir_path, exist_ok=True)
    log_file_path = os.path.join(log_dir_path, log_file_name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on multiple imports
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # File Handler with rotation --> keeps logs manageable
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger