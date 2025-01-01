"""Document analyzer model implementation."""

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer
from PIL import Image
import logging
from typing import List, Dict, Any
from utils.model_utils import (
    load_optimal_params,
    prepare_image,
    clean_model_output,
    optimize_memory
)
from config.settings import (
    MODEL_NAME,
    MIN_PIXELS,
    MAX_PIXELS,
    TARGET_IMAGE_SIZE,
    HYPERPARAMS_FILE,
    MAX_MEMORY
)

logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """Financial document analyzer using Qwen2-VL model."""
    
    def __init__(self, model, processor, device):
        """Initialize the analyzer with model components."""
        self.model = model
        self.processor = processor
        self.device = device
        self.best_params = load_optimal_params(HYPERPARAMS_FILE)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    @classmethod
    def create(cls, device):
        """Create a new analyzer instance."""
        try:
            processor = AutoProcessor.from_pretrained(
                MODEL_NAME,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS
            )

            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                max_memory=MAX_MEMORY
            ).to(device)

            return cls(model, processor, device)
        except Exception as e:
            logger.error(f"Error creating analyzer: {e}")
            raise

    def process_vision_info(self, messages: List[Dict[str, Any]]) -> tuple:
        """Extract images from messages."""
        image_list = []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image":
                    image_list.append(content["image"])
        return image_list, len(image_list)

    def prepare_inputs(self, image: Image.Image, query: str) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query}
            ]
        }]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, _ = self.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        device_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if 'pixel' in k:
                    device_inputs[k] = v.to(dtype=torch.float16, device=self.device)
                else:
                    device_inputs[k] = v.to(device=self.device, dtype=torch.long)
            else:
                device_inputs[k] = v
        
        return device_inputs

    def generate_analysis_prompt(self, page_num: int, total_pages: int) -> str:
        """Generate the analysis prompt."""
        return f"""Analyze this financial document page {page_num} of {total_pages} and provide a detailed analysis. Focus on:
    1.  Product name
    2.  Product type
    3.  ISIN identifier
    4.  Issuer and guarantor
    5.  Important dates (issue date, maturity date)
    6.  Protection barrier and underlying indices
    7.  Expected performance (scenarios: stress, unfavorable, moderate, favorable)
    8.  Risk indicator (scale 1-7)
    9.  Total and entry costs
    10. Target audience (investment horizon, risk tolerance)
    11. Product currency

If an element is not found on a page, indicate 'not present'. Summarize your findings in a clear table for each page"""

    def analyze_image(self, image: Image.Image, page_num: int, total_pages: int) -> str:
        """Analyze a single image."""
        try:
            image = prepare_image(image, TARGET_IMAGE_SIZE)
            query = self.generate_analysis_prompt(page_num, total_pages)
            inputs = self.prepare_inputs(image, query)

            with torch.inference_mode():
                output = self.model.generate(**inputs, **self.best_params)
                raw_text = self.processor.batch_decode(output.cpu(), skip_special_tokens=True)[0]
                cleaned_text = clean_model_output(raw_text)

                del output
                del inputs
                optimize_memory()

                return cleaned_text

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
