"""Application configuration settings."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
HYPERPARAMS_FILE = os.path.join(BASE_DIR, "optimal_hyperparams.json")

# Model settings
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
TARGET_IMAGE_SIZE = (800, 800)

# Memory settings
MAX_MEMORY = {'mps': '8GB'}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['default'],
        'level': 'INFO',
    }
}
