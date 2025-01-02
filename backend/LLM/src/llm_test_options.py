import logging
import json
import llama_cpp
import os
from typing import Dict, Any, List
from dataclasses import dataclass
from validation_advanced import validate_document
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_config() -> Dict[str, Any]:
    """Charge la configuration depuis config.json."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        raise

@dataclass
class LLMConfig:
    """Configuration pour le modèle LLM."""
    max_retries: int = 3
    temperature: float = 0.1
    max_tokens: int = 1000

def read_vlm_output(file_path: str) -> str:
    """Lit le contenu du fichier VLM."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            logger.info(f"Fichier {file_path} lu avec succès")
            return content
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier: {str(e)}")
        return ""

def save_json_output(data: Dict[str, Any], output_file: str) -> None:
    """Sauvegarde les données au format JSON après validation."""
    try:
        # Validation avancée
        validation_result = validate_document(data)
        
        # Affichage des résultats de validation
        logger.info(f"\nScore de qualité du document : {validation_result.score:.2%}")
        
        # Si le score est supérieur à 0, on considère que c'est valide
        if validation_result.score > 0:
            # Sauvegarde du JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"Résultats sauvegardés dans {output_file}")
            
            # Affichage du feedback
            if validation_result.feedback:
                logger.info("\nFeedback de validation:")
                for fb in validation_result.feedback:
                    logger.info(f"- {fb}")
        else:
            logger.warning("Document invalide - score trop bas")
            
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du JSON: {str(e)}")

def load_json_schema(schema_file: str) -> Dict[str, Any]:
    """Charge le schéma JSON depuis un fichier."""
    try:
        with open(schema_file, 'r', encoding='utf-8') as file:
            schema = json.load(file)
            logger.info(f"Schéma JSON chargé depuis {schema_file}")
            return schema
    except Exception as e:
        logger.error(f"Erreur lors du chargement du schéma JSON: {str(e)}")
        return {}

def main():
    try:
        # Chargement de la configuration
        config = load_config()
        model_config = config["model"]
        
        # Initialisation du modèle LLM
        llm = llama_cpp.Llama(
            model_path=model_config["path"],
            n_ctx=model_config["max_length"],
            n_threads=8,  # Utiliser plus de threads CPU
            n_batch=1024,  # Augmenter la taille du batch comme dans votre exemple
            n_gpu_layers=-1,  # Charger tous les layers sur le GPU
            use_mmap=True,  # Utiliser le memory mapping pour un chargement plus rapide
            use_mlock=False,  # Désactiver le verrouillage mémoire
            verbose=True  # Activer les logs pour voir ce qui se passe
        )
        
        # Lecture du fichier d'entrée
        input_path = os.path.join(project_root, "inputs", "vlm_output.txt")
        vlm_output = read_vlm_output(input_path)
        if not vlm_output:
            logger.error("Impossible de lire le fichier d'entrée")
            return

        # Chargement du schéma JSON
        schema_path = os.path.join(project_root, "configs", "json_schema.json")
        json_structure = load_json_schema(schema_path)
        if not json_structure:
            logger.error("Impossible de charger le schéma JSON")
            return
        
        # Préparation du prompt
        prompt = f"""[INST] You are a financial document analyzer. Extract information from the document below into the specified JSON format.

Rules:
1. Return ONLY the JSON object, no other text
2. The JSON must exactly match the provided structure
3. ALL dates must be in YYYY-MM-DD format
4. ALL percentages must be converted to decimals (e.g., -18.9% -> -0.189)
5. ALL monetary values should be numbers without currency symbols
6. Risk levels should be "Low" (1-2), "Medium" (3-4), or "High" (5-7)
7. If a value is not found in the document, use null
8. Arrays (factors, warnings) must not be empty if information exists
9. For risk factors, include:
   - Investment horizon (e.g., "5 year investment horizon")
   - Risk level description
   - Any other relevant risk information
10. For performance scenarios:
    - 'initial' should be the investment amount
    - 'final' should be the "what you might get back" amount
    - Convert all percentages to decimals

Document to analyze:
{vlm_output}

Required JSON structure:
{json.dumps(json_structure, indent=2)}

IMPORTANT: Return ONLY valid JSON, no other text. Make sure all JSON is properly formatted with correct commas and brackets.
[/INST]"""
        
        # Génération de la réponse
        response = llm(
            prompt,
            max_tokens=2048,  # Augmenter la limite de tokens
            temperature=model_config["temperature"],
            stop=["```", "[/INST]"],
            echo=False
        )
        
        # Log de la réponse brute
        logger.info("=== Réponse brute du LLM ===")
        logger.info(response["choices"][0]["text"])
        logger.info("=== Fin de la réponse brute ===")
        
        # Nettoyage et parsing de la réponse
        response_text = response["choices"][0]["text"].strip()
        
        try:
            # Essayer de compléter le JSON si nécessaire
            if not response_text.endswith("}"):
                logger.info("JSON incomplet détecté, tentative de réparation...")
                # Compter les accolades ouvrantes et fermantes
                open_braces = response_text.count("{")
                close_braces = response_text.count("}")
                
                if open_braces > close_braces:
                    # Ajouter les accolades manquantes
                    missing_braces = open_braces - close_braces
                    response_text += "}" * missing_braces
                    logger.info(f"Ajout de {missing_braces} accolade(s) fermante(s)")
            
            # Supprimer tout ce qui n'est pas entre les accolades
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            # Log après nettoyage
            logger.info("=== Réponse après nettoyage et réparation ===")
            if start >= 0 and end > start:
                response_text = response_text[start:end]
                logger.info(response_text)
            else:
                logger.error("Pas trouvé de JSON valide dans la réponse")
                return
            logger.info("=== Fin de la réponse nettoyée ===")
            
            # Première tentative de parsing JSON
            logger.info("Tentative de parsing JSON...")
            output_data = json.loads(response_text)
            logger.info("Parsing JSON réussi!")
            
            # Validation du document
            logger.info("Validation du document...")
            validation_result = validate_document(output_data)
            logger.info(f"Score de validation: {validation_result.score}")
            if validation_result.feedback:
                logger.info("Retours de validation:")
                for feedback in validation_result.feedback:
                    logger.info(f"- {feedback}")
            
            # Ne sauvegarder que si le score de validation est acceptable
            if validation_result.score >= 0.8:  # Seuil de 80%
                # Créer le dossier outputs s'il n'existe pas
                output_dir = os.path.join(project_root, "outputs")
                os.makedirs(output_dir, exist_ok=True)
                
                # Sauvegarder le résultat
                output_file = os.path.join(output_dir, "analysis_result.json")
                save_json_output(output_data, output_file)
                logger.info("Analyse terminée avec succès")
            else:
                raise ValueError(f"Score de validation trop bas: {validation_result.score}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur lors du parsing JSON : {str(e)}\nRéponse reçue : {response_text}")
            raise
            
    except Exception as e:
        logger.error(f"Erreur lors du traitement : {str(e)}")
        raise

if __name__ == "__main__":
    main()
