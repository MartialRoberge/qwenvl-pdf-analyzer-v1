"""Advanced document validation module."""

from typing import List, Dict, Any
import logging
import json
import os

logger = logging.getLogger(__name__)

class ValidationResult:
    """Container for validation results."""
    def __init__(self, score: float = 0.0, feedback: List[str] = None):
        self.score = score
        self.feedback = feedback or []

class DocumentValidator:
    """Advanced document validator."""
    
    def __init__(self):
        """Initialize validator with schema."""
        schema_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'json_schema.json')
        try:
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
        except Exception as e:
            logger.error(f"Error loading schema: {str(e)}")
            self.schema = {}

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate document analysis results against JSON schema.
        
        Args:
            data: Document analysis data
            
        Returns:
            ValidationResult with score and feedback
        """
        try:
            feedback = []
            score = 0.0
            
            # Validate main sections
            for section in self.schema:
                if section not in data:
                    feedback.append(f"Missing section: {section}")
                    continue
                
                section_data = data[section]
                section_schema = self.schema[section]
                
                # Check fields in section
                for field, field_type in section_schema.items():
                    if field not in section_data:
                        feedback.append(f"Missing field '{field}' in section '{section}'")
                        continue
                        
                    value = section_data[field]
                    
                    # Handle nested objects
                    if isinstance(field_type, dict):
                        for subfield, subfield_type in field_type.items():
                            if subfield not in value:
                                feedback.append(f"Missing subfield '{subfield}' in '{section}.{field}'")
                    
                    # Handle arrays
                    elif isinstance(field_type, list):
                        if not isinstance(value, list):
                            feedback.append(f"Field '{section}.{field}' should be a list")
                    
                    # Handle basic types
                    elif field_type == "str" and not isinstance(value, str):
                        feedback.append(f"Field '{section}.{field}' should be a string")
                    elif field_type == "float" and not isinstance(value, (int, float)):
                        feedback.append(f"Field '{section}.{field}' should be a number")
                    elif field_type == "YYYY-MM-DD" and not isinstance(value, str):
                        feedback.append(f"Field '{section}.{field}' should be a date string")
            
            # Calculate score based on completeness
            total_fields = sum(len(section.keys()) for section in self.schema.values())
            valid_fields = total_fields - len(feedback)
            score = valid_fields / total_fields if total_fields > 0 else 0.0
            
            return ValidationResult(
                score=round(score, 2),
                feedback=feedback
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                score=0.0,
                feedback=[f"Validation failed: {str(e)}"]
            )

def validate_document(data: Dict[str, Any]) -> ValidationResult:
    """Main entry point for document validation."""
    validator = DocumentValidator()
    return validator.validate(data)
