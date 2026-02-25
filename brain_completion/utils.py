import logging
import torch

def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    return logger

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
