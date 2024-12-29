import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf2image import convert_from_bytes
import logging
import nltk
import subprocess
import os
import json
from models.analyzer import DocumentAnalyzer
from utils.model_utils import setup_device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# Initialize device and analyzer
device = setup_device()
analyzer = DocumentAnalyzer.create(device)

def write_to_llm_input(analyses):
    """Write the VLM analyses to the LLM input file."""
    backend_dir = Path(__file__).parent.parent  # Remonter au dossier backend
    llm_input_path = os.path.join(backend_dir, 'LLM', 'inputs', 'vlm_output.txt')
    os.makedirs(os.path.dirname(llm_input_path), exist_ok=True)  # Créer le dossier s'il n'existe pas
    content = "\n\n".join([f"Page {analysis['page']}:\n{analysis['content']}" for analysis in analyses])
    with open(llm_input_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return llm_input_path

def run_llm_analysis():
    """Run the LLM analysis script."""
    backend_dir = Path(__file__).parent.parent  # Remonter au dossier backend
    llm_script_path = os.path.join(backend_dir, 'LLM', 'src', 'llm_test_options.py')
    output_path = os.path.join(backend_dir, 'LLM', 'outputs', 'analysis_result.json')
    
    try:
        # Exécuter le script LLM
        result = subprocess.run(
            [sys.executable, llm_script_path], 
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("LLM script executed successfully")
        
        # Vérifier que le fichier de sortie existe
        if not os.path.exists(output_path):
            logger.error("Output file not found")
            return None
            
        # Lire le fichier JSON
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
                logger.info("JSON file read successfully")
                logger.info(f"JSON content: {json_content}")  # Log du contenu
                return json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error reading output file: {str(e)}")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"LLM script error: {e.output}")
        return None
    except Exception as e:
        logger.error(f"Error running LLM analysis: {str(e)}")
        return None

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
                result = analyzer.analyze_image(image, i + 1, len(images))
                analyses.append({
                    'page': i + 1,  
                    'content': result
                })
                logger.info(f"Page {i + 1} processed successfully")
            except Exception as e:
                logger.error(f"Error processing page {i + 1}: {str(e)}")
                return jsonify({'error': f'Error processing page {i + 1}'}), 500

        # Write VLM output to LLM input file
        write_to_llm_input(analyses)
        
        # Run LLM analysis
        llm_result = run_llm_analysis()
        if llm_result is None:
            return jsonify({'error': 'Error running LLM analysis'}), 500
            
        # Log du résultat avant de le renvoyer
        logger.info(f"Sending result to frontend: {json.dumps(llm_result, indent=2)}")
        return jsonify(llm_result)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)