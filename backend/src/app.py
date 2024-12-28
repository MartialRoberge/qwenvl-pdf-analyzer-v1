import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf2image import convert_from_bytes
import logging
import nltk
from src.models.analyzer import DocumentAnalyzer
from src.utils.model_utils import setup_device

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
                analyses.append({
                    'page': i + 1,  
                    'content': f"Error processing page: {str(e)}"
                })

        return jsonify({'analyses': analyses})

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Error processing PDF: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)