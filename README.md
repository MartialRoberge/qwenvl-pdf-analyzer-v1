# Financial Document Analyzer

Un analyseur de documents financiers utilisant le modèle Qwen2-VL pour extraire automatiquement les informations clés des documents PDF.

## 📋 Table des matières

- [Prérequis](#prérequis)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Démarrage](#démarrage)
- [Configuration](#configuration)
- [Optimisation du modèle](#optimisation-du-modèle)
- [Architecture détaillée](#architecture-détaillée)
- [Guide de dépannage](#guide-de-dépannage)

## 🔧 Prérequis

- Python 3.8+
- Node.js (pour servir le frontend)
- pip (gestionnaire de paquets Python)
- Navigateur web moderne
- Au moins 8GB de RAM
- Support MPS (Metal Performance Shaders) pour Mac ou CPU puissant

## 💻 Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/MartialRoberge/qwenvl-pdf-analyzer-v1.git
cd qwenvl-pdf-analyzer-v1
```

2. Installez les dépendances Python :
```bash
cd backend
pip install -r requirements.txt
```

## 📁 Structure du projet

```
qwenvl-pdf-analyzer/
├── backend/
│   ├── src/
│   │   ├── config/
│   │   │   └── settings.py      # Configuration globale
│   │   ├── models/
│   │   │   └── analyzer.py      # Classe principale d'analyse
│   │   ├── utils/
│   │   │   └── model_utils.py   # Fonctions utilitaires
│   │   └── app.py              # Point d'entrée de l'API
│   ├── optimize.py             # Script d'optimisation
│   └── requirements.txt        # Dépendances Python
├── frontend/
│   └── index.html             # Interface utilisateur
└── reference_analyses.json    # Données de référence pour l'optimisation
```

## 🚀 Démarrage

1. Démarrez le backend :
```bash
cd backend
python3 src/app.py
```
Le serveur démarrera sur http://localhost:5004

2. Pour le frontend, ouvrez simplement `frontend/index.html` dans votre navigateur ou utilisez un serveur web simple :
```bash
cd frontend
python3 -m http.server 8000
```
Puis accédez à http://localhost:8000

## ⚙️ Configuration

### Configuration du modèle (`settings.py`)

```python
# Taille des images
TARGET_IMAGE_SIZE = (800, 800)  # Augmentez pour plus de détails, diminuez pour la vitesse

# Limites de mémoire
MAX_MEMORY = {'mps': '8GB'}  # Ajustez selon votre matériel

# Paramètres du modèle
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"  # Version du modèle
```

### Paramètres d'analyse (`analyzer.py`)

La classe `DocumentAnalyzer` contient les paramètres de génération dans `best_params` :

```python
{
    'max_new_tokens': 1000,     # Longueur maximale de la réponse
    'do_sample': True,          # Génération avec échantillonnage
    'temperature': 0.7,         # Créativité (0.1-1.0)
    'top_p': 0.9,              # Filtrage des tokens (0.1-1.0)
    'top_k': 40,               # Nombre de meilleurs tokens
    'repetition_penalty': 1.3   # Pénalité pour les répétitions
}
```

## 🎯 Optimisation du modèle

Le script `optimize.py` permet d'optimiser automatiquement les paramètres du modèle.

### Comment ça marche

1. **Données de référence** : 
   - Créez un fichier `reference_analyses.json` avec vos analyses de référence
   - Format :
   ```json
   {
     "analyses": [
       {
         "page": 1,
         "expected_text": "Votre analyse de référence..."
       }
     ]
   }
   ```

2. **Lancer l'optimisation** :
```bash
cd backend
python3 optimize.py
```

3. **Paramètres d'optimisation** :
   - Le script teste différentes combinaisons de paramètres
   - Plages de valeurs testées :
   ```python
   'temperature': (0.1, 1.0)
   'top_p': (0.1, 1.0)
   'top_k': (1, 100)
   'repetition_penalty': (1.0, 2.0)
   'max_new_tokens': (100, 2000)
   ```

4. **Résultats** :
   - Les meilleurs paramètres sont sauvegardés dans `optimal_hyperparams.json`
   - Le modèle les utilisera automatiquement

### Personnalisation de l'optimisation

Pour modifier les plages de valeurs :
1. Ouvrez `optimize.py`
2. Modifiez les valeurs dans la fonction `objective` :
```python
params = {
    'temperature': trial.suggest_float('temperature', MIN, MAX),
    # ...
}
```

## 🏗 Architecture détaillée

### Backend

1. **app.py** :
   - Point d'entrée de l'API
   - Gère les requêtes HTTP
   - Route principale : `/analyze` (POST)

2. **analyzer.py** :
   - Classe `DocumentAnalyzer`
   - Gère l'analyse des images
   - Génère les prompts
   - Nettoie les sorties

3. **model_utils.py** :
   - Fonctions utilitaires
   - Gestion de la mémoire
   - Configuration du device

4. **settings.py** :
   - Configuration globale
   - Constantes
   - Paramètres du modèle

### Frontend

- Interface responsive
- Upload par glisser-déposer
- Animations de chargement
- Formatage Markdown des résultats
- Copie rapide des analyses

## 🔍 Guide de dépannage

### Problèmes courants

1. **Erreur de mémoire** :
   - Diminuez `TARGET_IMAGE_SIZE`
   - Réduisez `max_new_tokens`
   - Ajustez `MAX_MEMORY`

2. **Analyses imprécises** :
   - Augmentez `temperature` pour plus de créativité
   - Diminuez `temperature` pour plus de précision
   - Ajustez `top_p` et `top_k`

3. **Lenteur** :
   - Réduisez `TARGET_IMAGE_SIZE`
   - Diminuez `max_new_tokens`
   - Utilisez un device plus rapide

### Logs

- Les logs sont dans la console du backend
- Niveau INFO par défaut
- Modifiez le niveau dans `settings.py`

## 📝 Notes

- Le modèle fonctionne mieux avec des images de bonne qualité
- L'optimisation peut prendre du temps
- Les paramètres optimaux dépendent de vos données de référence

## 🤝 Contribution

1. Fork le projet
2. Créez votre branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request