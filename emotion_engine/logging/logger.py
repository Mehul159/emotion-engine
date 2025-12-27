import logging
import os

LOG_LEVEL = os.getenv("EMOTION_ENGINE_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
LOG_FILE = os.getenv("EMOTION_ENGINE_LOG_FILE", "emotion_engine.log")

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def get_logger(name: str):
    return logging.getLogger(name)
