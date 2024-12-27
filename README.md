# Financial Document Analyzer

A modern web application that analyzes financial documents using AI. Built with Python (Flask) backend and React (TypeScript) frontend.

## Features

- PDF document analysis with AI-powered text extraction
- Page-by-page analysis of financial documents
- Extraction of key financial information:
  - Product details (name, type, ISIN)
  - Important dates and terms
  - Performance scenarios
  - Risk indicators
  - Costs and fees
- Modern, responsive UI with drag-and-drop functionality
- Real-time analysis feedback

## Technical Stack

### Backend
- Python 3.12+
- Flask for API endpoints
- Qwen-VL for AI document analysis
- PyTorch with MPS acceleration for Apple Silicon
- PDF processing with pdf2image

### Frontend
- React 18 with TypeScript
- Vite for fast development
- Tailwind CSS for styling
- Framer Motion for animations
- Axios for API communication

## Setup

### Prerequisites
- Python 3.12+
- Node.js 16+
- pip
- npm

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## Usage

1. Start both backend and frontend servers
2. Open http://localhost:5173 in your browser
3. Drag and drop a financial PDF document or click to select one
4. Click "Analyze Document" to start the analysis
5. View the results page by page

## API Endpoints

- `POST /analyze`: Analyze a PDF document
  - Input: PDF file in multipart/form-data
  - Output: JSON with analysis results

## Development

- Backend runs on port 5004
- Frontend runs on port 5173
- MPS acceleration is automatically used on Apple Silicon Macs