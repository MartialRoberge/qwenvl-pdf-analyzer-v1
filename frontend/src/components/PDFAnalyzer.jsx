import React, { useState, useCallback } from 'react';
import './PDFAnalyzer.css';

const PDFAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [analysis, setAnalysis] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    setError('');

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile?.type !== 'application/pdf') {
      setError('Veuillez déposer un fichier PDF uniquement.');
      return;
    }
    setFile(droppedFile);
  }, []);

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile?.type !== 'application/pdf') {
      setError('Veuillez sélectionner un fichier PDF uniquement.');
      return;
    }
    setFile(selectedFile);
    setError('');
  };

const handleAnalyze = async () => {
    if (!file) return;

    setIsLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('pdf', file);

      // Ajout des logs pour debug
      console.log("Sending request to backend...");
      console.log("File being sent:", file);

      const response = await fetch('http://localhost:5004/analyze', {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json',
        },
        mode: 'cors',
        credentials: 'omit'
      });

      console.log("Response status:", response.status);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Erreur lors de l\'analyse du PDF');
      }

      const result = await response.json();
      setAnalysis(result.text);
    } catch (err) {
      console.error("Error details:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <div
        className={`dropzone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {file ? (
          <div className="file-info">
            <p>Fichier sélectionné : {file.name}</p>
            <button
              className="button"
              onClick={() => setFile(null)}
            >
              Supprimer
            </button>
          </div>
        ) : (
          <div>
            <p>Déposez votre fichier PDF ici ou</p>
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileSelect}
              className="file-input"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="button">
              Sélectionnez un fichier
            </label>
          </div>
        )}
      </div>

      {error && (
        <div className="error">
          {error}
        </div>
      )}

      <button
        className={`button analyze-button ${isLoading ? 'loading' : ''}`}
        onClick={handleAnalyze}
        disabled={!file || isLoading}
      >
        {isLoading ? 'Analyse en cours...' : 'Analyser le PDF'}
      </button>

      {analysis && (
        <div className="analysis-result">
          <h3>Résultat de l'analyse :</h3>
          <div className="result-text">
            {analysis}
          </div>
        </div>
      )}
    </div>
  );
};

export default PDFAnalyzer;