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
1. Extract information EXACTLY as it appears in the document
2. Do not modify, round, or reformat any numbers or text
3. Only include information from the current page
4. If information is not found, write "Not found on this page"
5. Do not make assumptions or inferences

For example:
- If you see "EUR 10,000", write exactly "EUR 10,000" (not "10000 EUR" or "10,000 EUR")
- If you see "87.50%", write exactly "87.50%" (not "87.5%" or "88%")
- If you see "03 May 2024", write exactly "03 May 2024" (not "3/5/24" or "2024-05-03")

Extract and format the information as follows:

PAGE 1 STRUCTURE:
A. Product Details:
   - Product Name: [copy exact name]
   - ISIN: [copy exact code]
   - Manufacturer: [copy exact name]
   - Product Type: [copy exact description]
   - Issue Date: [copy exact date]
   - Maturity Date: [copy exact date]
   - Currency: [copy exact code]
   - Nominal Amount: [copy exact amount with format]

B. Product Structure:
   - Payment Conditions: [copy exact conditions with percentages]
   - Protection Level: [copy exact percentage]
   - Underlying Index: [copy exact name and code]

C. Investor Profile:
   - Investment Horizon: [copy exact period]
   - Risk Tolerance: [copy exact percentage or description]
   - Required Knowledge: [copy exact description]

PAGE 2 STRUCTURE:
A. Risk Assessment:
   - Risk Level: [copy exact number/7]
   - Risk Factors: [list each factor exactly as written]

B. Performance Scenarios (€10,000 investment):
   Stress Scenario:
   - 1 year: [copy exact amount] ([copy exact percentage])
   - Maturity: [copy exact amount] ([copy exact percentage])
   
   Unfavorable Scenario:
   - 1 year: [copy exact amount] ([copy exact percentage])
   - Maturity: [copy exact amount] ([copy exact percentage])
   
   Moderate Scenario:
   - 1 year: [copy exact amount] ([copy exact percentage])
   - Maturity: [copy exact amount] ([copy exact percentage])
   
   Favorable Scenario:
   - 1 year: [copy exact amount] ([copy exact percentage])
   - Maturity: [copy exact amount] ([copy exact percentage])

PAGE 3 STRUCTURE:
A. Cost Structure:
   One-off Costs:
   - Entry: [copy exact percentage]
   - Exit: [copy exact percentage]
   
   Ongoing Costs:
   - Transaction: [copy exact percentage]
   - Other: [copy exact percentage]
   
   Incidental Costs:
   - Performance Fees: [copy exact percentage]
   - Carried Interest: [copy exact percentage]

B. Total Cost Impact:
   - After 1 year: [copy exact amount] ([copy exact percentage])
   - At maturity: [copy exact amount] ([copy exact percentage])

C. Holding Information:
   - Recommended Period: [copy exact duration]
   - Early Exit: [copy exact conditions]
   - Secondary Market: [copy exact details]

IMPORTANT: Your response should be a direct copy of the information as it appears in the document. Do not interpret, summarize, or modify any values."""

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
