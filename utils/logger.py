import logging
import os
from datetime import datetime

def setup_logger(name, log_dir="logs"):
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    
    return logger
