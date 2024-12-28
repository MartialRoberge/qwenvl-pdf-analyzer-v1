"""Document analyzer model implementation."""

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer
from PIL import Image
import logging
from typing import List, Dict, Any
from src.utils.model_utils import (
    load_optimal_params,
    prepare_image,
    clean_model_output,
    optimize_memory
)
from src.config.settings import (
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
        base_prompt = f"""You are a specialized financial document analyzer for Key Information Documents (KID). Your task is to analyze page {page_num} of {total_pages} of this KID document.

        CRITICAL INSTRUCTIONS:
        1. Extract information EXACTLY as it appears - preserve all formatting, spacing, and punctuation
        2. Never modify numbers, dates, currencies, or percentages
        3. Only include information visible on the current page
        4. If information is not found, state "Not found on this page"
        5. Do not make assumptions or inferences
        6. Keep exact currency formats (e.g., "EUR", "GBP")
        7. Maintain all numerical values exactly as shown
        
        PAGE 1 STRUCTURE:
        Product Identification:
        - Product Name: [exact name with all codes/references]
        - ISIN: [exact code]
        - Additional IDs: [Valor/Series number if present]
        - Manufacturer: [exact name]
        - Regulatory Status: [exact regulatory information]
        - Production Date/Time: [exact timestamp if shown]
        
        Product Terms:
        - Product Type: [exact description]
        - Issue Date: [exact date]
        - Maturity Date: [exact date]
        - Currency: [exact code]
        - Nominal Amount: [exact amount with currency]
        
        Underlying Assets:
        - Names: [exact names/codes of all indices/assets]
        - Initial Levels: [exact values if shown]
        - Strike Levels: [exact percentages]
        - Barrier Levels: [exact percentages]
        - Autocall Levels: [exact percentages]
        
        Investment Mechanics:
        - Objective: [exact description]
        - Payoff Structure: [exact conditions]
        - Early Termination: [exact autocall conditions]
        - Protection Features: [exact description]
        
        PAGE 2 STRUCTURE:
        Risk Assessment:
        - Risk Indicator: [exact number/7]
        - Risk Factors: [list each exactly as written]
        - Market Impact: [exact description]
        
        Performance Scenarios:
        Initial Investment: [exact amount]
        
        For each period ([list all shown periods]):
        1. Stress Scenario:
           - Amount: [exact with currency]
           - Return: [exact with +/- percentage]
        
        2. Unfavorable Scenario:
           - Amount: [exact with currency]
           - Return: [exact with +/- percentage]
        
        3. Moderate Scenario:
           - Amount: [exact with currency]
           - Return: [exact with +/- percentage]
        
        4. Favorable Scenario:
           - Amount: [exact with currency]
           - Return: [exact with +/- percentage]
        
        PAGE 3 STRUCTURE:
        Cost Structure:
        One-off Costs:
        - Entry: [exact percentage]
        - Exit: [exact percentage]
        
        Ongoing Costs:
        - Portfolio Transaction: [exact percentage]
        - Management Fees: [exact percentage]
        - Other: [exact percentage]
        
        Incidental Costs:
        - Performance Fees: [exact percentage]
        - Carried Interest: [exact percentage]
        
        Total Cost Impact:
        For each period:
        - Duration: [exact period]
        - Amount: [exact with currency]
        - Impact: [exact percentage]
        
        Additional Information:
        - Recommended Holding: [exact duration]
        - Early Exit Terms: [exact conditions]
        - Secondary Market: [exact details]
        - Complaint Process: [exact procedure]
        
        REMINDERS:
        1. Keep exact currency formats (EUR, GBP)
        2. Preserve all separators (e.g., "10,000.00")
        3. Maintain exact percentage formats (e.g., "+87.50%", "-12.50%")
        4. Copy dates exactly as written
        5. Include all regulatory information
        6. Preserve any reference codes exactly"""
        
        return base_prompt

    def analyze_image(self, image: Image.Image, page_num: int, total_pages: int) -> str:
        """Analyze a single image."""
        try:
            image = prepare_image(image, TARGET_IMAGE_SIZE)
            query = self.generate_analysis_prompt(page_num, total_pages)
            inputs = self.prepare_inputs(image, query)

            default_params = {
                'max_new_tokens': 1024,
                'temperature': 0.7,
                'do_sample': True,
                'top_p': 0.9,
                'num_beams': 1
            }

            with torch.inference_mode():
                output = self.model.generate(**inputs, **default_params)
                raw_text = self.processor.batch_decode(output.cpu(), skip_special_tokens=True)[0]
                cleaned_text = clean_model_output(raw_text)

                del output
                del inputs
                optimize_memory()

                return cleaned_text

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
