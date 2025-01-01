"""Utility functions for model operations."""

import torch
import logging
from typing import Dict, Any, Tuple
from PIL import Image
import json
from pathlib import Path
import re
import numpy as np

logger = logging.getLogger(__name__)

def setup_device() -> torch.device:
    """Configure and return the processing device."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

def load_optimal_params(hyperparams_file: str) -> Dict[str, Any]:
    """Load optimal parameters from file."""
    try:
        if Path(hyperparams_file).exists():
            with open(hyperparams_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load optimal parameters: {e}")
    
    return {
        'max_new_tokens': 1000,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 40,
        'repetition_penalty': 1.3
    }

def prepare_image(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Prepare image for model input."""
    if isinstance(image, Image.Image):
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Resize
        return image.resize(target_size, Image.Resampling.LANCZOS)
    return image

def clean_model_output(text: str) -> str:
    """Clean model output text."""
    if 'assistant' in text:
        text = text.split('assistant')[-1].strip()
    
    # Remove page mentions already included in interface
    text = re.sub(r'^Page \d+ of \d+\s*', '', text, flags=re.MULTILINE)
    
    # Remove extra spaces and newlines
    text = re.sub(r'\n\s*\n', '\n\n', text.strip())
    
    return text

def optimize_memory():
    """Optimize memory usage."""
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
