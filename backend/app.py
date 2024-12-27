from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf2image import convert_from_bytes
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
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
TARGET_IMAGE_SIZE = (1024, 1024)
HYPERPARAMS_FILE = "optimal_hyperparams.json"

class ModelOptimizer:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.best_params = self.load_optimal_params()

    def load_optimal_params(self):
        """Load optimal parameters if they exist"""
        try:
            if Path(HYPERPARAMS_FILE).exists():
                with open(HYPERPARAMS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load optimal parameters: {e}")
        
        return {
            'max_new_tokens': 500,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.2,
            'num_beams': 4,
            'length_penalty': 1.0
        }

    def optimize(self, reference_data):
        """
        Optimize model parameters using reference data
        reference_data: list of tuples (image, expected_text)
        """
        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                'max_new_tokens': trial.suggest_int('max_new_tokens', 200, 1000),
                'temperature': trial.suggest_float('temperature', 0.1, 1.0),
                'top_p': trial.suggest_float('top_p', 0.1, 1.0),
                'top_k': trial.suggest_int('top_k', 10, 100),
                'repetition_penalty': trial.suggest_float('repetition_penalty', 1.0, 2.0),
                'num_beams': trial.suggest_int('num_beams', 1, 8),
                'length_penalty': trial.suggest_float('length_penalty', 0.1, 2.0)
            }
            
            scores = []
            for image, expected_text in reference_data:
                try:
                    # Generate text with current parameters
                    inputs = self.processor(
                        images=image,
                        text=generate_analysis_prompt(1, 1),
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            **params,
                            do_sample=True
                        )
                    
                    generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    
                    # Calculate BLEU score
                    reference = [expected_text.split()]
                    candidate = generated_text.split()
                    bleu_score = sentence_bleu(reference, candidate)
                    
                    # Add other metrics as needed
                    content_score = self.evaluate_content_quality(generated_text, expected_text)
                    
                    # Combine scores
                    final_score = (bleu_score * 0.6) + (content_score * 0.4)
                    scores.append(final_score)
                
                except Exception as e:
                    logger.error(f"Error in optimization trial: {e}")
                    scores.append(0.0)
            
            return sum(scores) / len(scores)

        # Create and run optimization study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        
        # Save optimal parameters
        self.best_params = study.best_params
        with open(HYPERPARAMS_FILE, 'w') as f:
            json.dump(study.best_params, f)
        
        return study.best_params

    def evaluate_content_quality(self, generated_text, expected_text):
        """
        Evaluate the quality of generated content
        Add your custom metrics here
        """
        # Example metrics:
        # 1. Length ratio
        length_ratio = min(len(generated_text) / len(expected_text), 1.0)
        
        # 2. Key information presence
        key_info_score = self.check_key_information(generated_text, expected_text)
        
        # 3. Structure similarity
        structure_score = self.evaluate_structure(generated_text, expected_text)
        
        # Combine scores
        return (length_ratio + key_info_score + structure_score) / 3

    def check_key_information(self, generated_text, expected_text):
        """Check if key information is present in generated text"""
        key_elements = [
            # Product identification
            r'ISIN:\s*XS1914695009',
            r'BNP Paribas',
            r'87\.50%\s*Protection',
            
            # Dates and amounts
            r'03 May 2024',
            r'EUR 1,000',
            r'EUR 10,000',
            
            # Percentages and numbers
            r'\d+\.\d+%',
            r'3 out of 7',
            
            # Financial terms
            r'Initial Reference Price',
            r'Final Reference Price',
            r'Notional Amount',
            
            # Risk and performance
            r'stress scenario',
            r'favorable scenario',
            r'unfavorable scenario',
            
            # Costs
            r'Entry costs',
            r'Exit costs',
            r'ongoing costs'
        ]
        
        score = 0
        for element in key_elements:
            if re.search(element, generated_text, re.IGNORECASE):
                score += 1
        
        return score / len(key_elements)

    def evaluate_structure(self, generated_text, expected_text):
        """Evaluate the structural similarity of the texts"""
        sections = [
            r'PRODUCT\s+DATA',
            r'RISK\s+INDICATOR',
            r'PERFORMANCE\s+SCENARIOS',
            r'COSTS?\s+OVER\s+TIME',
            r'COMPOSITION\s+OF\s+COSTS'
        ]
        
        # Check section presence and order
        score = 0
        last_pos = -1
        for section in sections:
            match = re.search(section, generated_text, re.IGNORECASE)
            if match:
                score += 1
                current_pos = match.start()
                if current_pos > last_pos:
                    score += 0.5  # Bonus for correct order
                last_pos = current_pos
        
        return score / (len(sections) * 1.5)  # Normalize score

def setup_device():
    """Configure the processing device (MPS or CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS for computation")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for computation")
    return device

def load_model(device):
    """Load and configure the model and processor"""
    try:
        # Initialize processor
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS
        )
        
        # Initialize model
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16
        ).to(device)
        
        return processor, model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def process_image(image):
    """Process a single image"""
    try:
        # Resize image maintaining aspect ratio
        image.thumbnail(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def generate_analysis_prompt(page_num, total_pages):
    """Generate the analysis prompt for the model"""
    return f"""
    You are analyzing a financial document (page {page_num} of {total_pages}).
    Please follow these instructions strictly:
    1. Identify and extract only the information explicitly present on this page.
    2. For costs, risk indicators, and performance data, provide exact details.
    3. Enumerate all tables found and summarize their contents.
    4. Validate all ISIN numbers and identifiers against their standard format.
    5. If information is not present, indicate "Not Found" rather than making assumptions.
    """

# Initialize device, model and optimizer
device = setup_device()
processor, model = load_model(device)
model_optimizer = ModelOptimizer(model, processor, device)

@app.route('/optimize', methods=['POST'])
def optimize_model():
    """Endpoint to optimize model parameters using reference data"""
    try:
        if 'pdf' not in request.files or 'reference' not in request.files:
            return jsonify({'error': 'Both PDF and reference files are required'}), 400

        # Process input PDF
        pdf_file = request.files['pdf']
        pdf_bytes = pdf_file.read()
        images = convert_from_bytes(pdf_bytes)
        processed_images = [process_image(image) for image in images]

        # Load reference texts
        reference_file = request.files['reference']
        reference_texts = json.loads(reference_file.read().decode('utf-8'))

        # Create reference data pairs
        reference_data = list(zip(processed_images, reference_texts))

        # Run optimization
        optimal_params = model_optimizer.optimize(reference_data)

        return jsonify({
            'status': 'success',
            'optimal_parameters': optimal_params
        })

    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    """Endpoint to analyze PDF documents using optimized parameters"""
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400

        pdf_file = request.files['pdf']
        pdf_bytes = pdf_file.read()

        images = convert_from_bytes(pdf_bytes)
        results = []

        for page_num, image in enumerate(images, 1):
            logger.info(f"Processing page {page_num}/{len(images)}")
            
            # Process image
            processed_image = process_image(image)
            
            # Generate prompt
            query = generate_analysis_prompt(page_num, len(images))
            
            # Prepare inputs
            inputs = processor(
                images=processed_image,
                text=query,
                return_tensors="pt"
            ).to(device)

            # Generate output
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **model_optimizer.best_params,
                    do_sample=True
                )
            
            response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            results.append({
                'page': page_num,
                'analysis': response
            })

        return jsonify({
            'status': 'success',
            'results': results
        })

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)