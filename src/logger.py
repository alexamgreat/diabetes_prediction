import logging
import os
from datetime import datetime
from pathlib import Path



LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.dirname(logs_path), exist_ok=True)   



LOG_FILE_PATH = os.path.join("logs", LOG_FILE)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(levelname)s - %(message)s",
    level=logging.INFO,
    
    )


   