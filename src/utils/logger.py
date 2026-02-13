import logging
import sys
from pathlib import Path
from config.paths import LOGS_DIR


def setup_logger(name="JetGuard", log_file="system.log", level=logging.INFO):
    """
    Sets up a centralized logger that outputs to both console and file.
    """

    # Create Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers are already added (to avoid duplicate logs in Streamlit)
    if logger.hasHandlers():
        return logger

    # 1. File Handler (Saves to file)
    file_handler = logging.FileHandler(LOGS_DIR / log_file)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # 2. Stream Handler (Prints to console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_format = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_format)

    # Add Handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


# Global instance to be used everywhere
logger = setup_logger()