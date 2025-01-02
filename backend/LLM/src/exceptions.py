"""Exceptions personnalisées pour le projet."""

class ValidationError(Exception):
    """Exception levée pour les erreurs de validation."""
    def __init__(self, message: str, errors: list = None):
        super().__init__(message)
        self.errors = errors or []

class LLMProcessingError(Exception):
    """Exception levée pour les erreurs de traitement LLM."""
    def __init__(self, message: str, response: str = None):
        super().__init__(message)
        self.response = response

class ConfigurationError(Exception):
    """Exception levée pour les erreurs de configuration."""
    pass

class FileOperationError(Exception):
    """Exception levée pour les erreurs de manipulation de fichiers."""
    def __init__(self, message: str, file_path: str):
        super().__init__(f"{message}: {file_path}")
        self.file_path = file_path
