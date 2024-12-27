from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf2image import convert_from_bytes
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer
import torch
import logging
import optuna
from PIL import Image
import io
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# Constants
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
TARGET_IMAGE_SIZE = (480, 480)
HYPERPARAMS_FILE = "optimal_hyperparams.json"

def setup_device():
    """Configure the processing device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

def load_model(device):
    """Load and configure the model and processor"""
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
            max_memory={'mps': '8GB'}  # Limite la mémoire MPS
        ).to(device)

        return processor, model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

class ModelOptimizer:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.best_params = self.load_optimal_params()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def load_optimal_params(self):
        """Load optimal parameters if they exist"""
        try:
            if Path(HYPERPARAMS_FILE).exists():
                with open(HYPERPARAMS_FILE, 'r') as f:
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

    def process_vision_info(self, messages):
        """Extract images from messages"""
        image_list = []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image":
                    image_list.append(content["image"])
        return image_list, len(image_list)

    def prepare_inputs(self, image, query):
        """Prepare inputs for the model"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query}
            ]
        }]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Get images
        image_inputs, _ = self.process_vision_info(messages)

        # Prepare inputs with memory optimization
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            truncation=True,  # Tronque si nécessaire
            return_tensors="pt"
        )

        # Convertir uniquement les tenseurs d'image en float16, garder les indices en long
        device_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if 'pixel' in k:  # Pour les tenseurs d'image
                    device_inputs[k] = v.to(dtype=torch.float16, device=self.device)
                else:  # Pour les indices et masques
                    device_inputs[k] = v.to(device=self.device, dtype=torch.long)
            else:
                device_inputs[k] = v
        
        return device_inputs

    def clean_model_output(self, text: str) -> str:
        """Nettoie la sortie du modèle pour ne garder que la réponse pertinente"""
        # Enlève le préfixe du système et de l'utilisateur
        if 'assistant' in text:
            text = text.split('assistant')[-1].strip()
        
        # Enlève les mentions de page qui sont déjà incluses dans l'interface
        text = re.sub(r'^Page \d+ of \d+\s*', '', text, flags=re.MULTILINE)
        
        # Enlève les espaces et sauts de ligne en trop
        text = re.sub(r'\n\s*\n', '\n\n', text.strip())
        
        return text

    def generate_analysis_prompt(self, page_num, total_pages):
        """Generate the analysis prompt"""
        return f"""Analyze this financial document page {page_num} of {total_pages} and provide a detailed analysis. Focus on:
	1.	Product name
	2.	Product type
	3.	ISIN identifier
	4.	Issuer and guarantor
	5.	Important dates (issue date, maturity date)
	6.	Protection barrier and underlying indices
	7.	Expected performance (scenarios: stress, unfavorable, moderate, favorable)
	8.	Risk indicator (scale 1-7)
	9.	Total and entry costs
	10.	Target audience (investment horizon, risk tolerance)
	11.	Product currency

If an element is not found on a page, indicate ‘not present’. Summarize your findings in a clear table for each page"""

    def analyze_image(self, image, page_num, total_pages):
        """Analyze a single image"""
        try:
            # Resize image
            if isinstance(image, Image.Image):
                image = image.resize(TARGET_IMAGE_SIZE)

            # Generate query
            query = self.generate_analysis_prompt(page_num, total_pages)

            # Prepare inputs
            inputs = self.prepare_inputs(image, query)

            # Generate text with memory optimization
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    **self.best_params,
                    do_sample=True
                )
                
                # Décode et nettoie la sortie
                raw_text = self.processor.batch_decode(output.cpu(), skip_special_tokens=True)[0]
                cleaned_text = self.clean_model_output(raw_text)
                
                # Libère la mémoire explicitement
                del output
                del inputs
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                
                return cleaned_text

        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise

@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400

        pdf_file = request.files['pdf']
        pdf_bytes = pdf_file.read()

        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes)
        analyses = []

        for i, image in enumerate(images):
            logger.info(f"Processing page {i + 1}")
            
            try:
                # Analyze the image
                result = model_optimizer.analyze_image(image, i + 1, len(images))
                analyses.append({
                    'page_number': i + 1,
                    'content': result
                })
                logger.info(f"Page {i + 1} processed successfully")
            
            except Exception as e:
                logger.error(f"Error processing page {i + 1}: {str(e)}")
                analyses.append({
                    'page_number': i + 1,
                    'content': f"Error processing page: {str(e)}"
                })

        return jsonify({'analyses': analyses})

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Error processing PDF: {str(e)}"}), 500

# Initialize device, model and optimizer
device = setup_device()
processor, model = load_model(device)
model_optimizer = ModelOptimizer(model, processor, device)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)