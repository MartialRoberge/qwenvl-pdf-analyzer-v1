import json
import optuna
import torch
from app import ModelOptimizer, setup_device, load_model
from pathlib import Path
from pdf2image import convert_from_path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_reference_data():
    """Load reference analyses"""
    with open('../reference_analyses.json', 'r') as f:
        data = json.load(f)
        return data['analyses']

def calculate_simple_similarity(ref_text, gen_text):
    """Calculate a simple word overlap score"""
    ref_words = set(ref_text.lower().split())
    gen_words = set(gen_text.lower().split())
    
    if not gen_words:  # Si le texte généré est vide
        return 0.0
    
    common_words = ref_words.intersection(gen_words)
    return len(common_words) / max(len(ref_words), len(gen_words))

def objective(trial):
    """Optimization objective function"""
    # Paramètres à optimiser
    params = {
        'temperature': trial.suggest_float('temperature', 0.1, 1.0),
        'top_p': trial.suggest_float('top_p', 0.1, 1.0),
        'top_k': trial.suggest_int('top_k', 1, 100),
        'repetition_penalty': trial.suggest_float('repetition_penalty', 1.0, 2.0),
        'max_new_tokens': trial.suggest_int('max_new_tokens', 100, 2000)
    }
    
    logger.info(f"\nTesting parameters: {params}")
    
    # Charger les données de référence et le PDF
    reference_data = load_reference_data()
    pdf_pages = convert_from_path('../test.pdf')
    
    if not pdf_pages:
        logger.error("Could not load PDF pages")
        return 0.0
    
    # Initialiser le modèle avec les nouveaux paramètres
    device = setup_device()
    processor, model = load_model(device)
    optimizer = ModelOptimizer(model, processor, device)
    optimizer.best_params = params
    
    total_scores = []
    
    # Évaluer sur chaque exemple de référence
    for ref in reference_data:
        try:
            page_idx = ref['page'] - 1
            if page_idx < len(pdf_pages):
                # Générer une analyse avec les paramètres actuels
                generated = optimizer.analyze_image(pdf_pages[page_idx], ref['page'], len(reference_data))
                
                # Log des textes pour comparaison
                logger.info(f"\n=== Page {ref['page']} ===")
                logger.info("\nRéférence:")
                logger.info(ref['expected_text'][:500])
                logger.info("\nGénéré:")
                logger.info(generated[:500])
                
                # Calculer un score de similarité simple
                score = calculate_simple_similarity(ref['expected_text'], generated)
                
                total_scores.append(score)
                logger.info(f"\nScore de similarité: {score}")
                
            else:
                logger.warning(f"Page {ref['page']} not found in PDF")
            
        except Exception as e:
            logger.error(f"Error processing reference: {e}", exc_info=True)
            return 0.0
    
    # Libérer la mémoire
    del model
    del processor
    del optimizer
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    final_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
    logger.info(f"\nTrial final score: {final_score}")
    return final_score

def optimize():
    """Run the optimization process"""
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    # Sauvegarder les meilleurs paramètres
    best_params = study.best_params
    best_params['do_sample'] = True
    
    with open('optimal_hyperparams.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score: {study.best_value}")

if __name__ == '__main__':
    optimize()
