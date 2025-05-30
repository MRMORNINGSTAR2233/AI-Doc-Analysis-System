import os
import google.generativeai as genai
from pathlib import Path
import json
import logging
from typing import Dict, Any, List
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

class JSONAgent:
    def __init__(self):
        # Initialize Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Common JSON schemas
        self.schemas = {
            "transaction": {
                "type": "object",
                "required": ["amount", "currency", "timestamp"],
                "properties": {
                    "amount": {"type": "number"},
                    "currency": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string"}
                }
            },
            "user_profile": {
                "type": "object",
                "required": ["id", "name", "email"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "preferences": {"type": "object"}
                }
            }
        }
        
    def _validate_schema(self, data: Dict[str, Any], schema_type: str) -> List[str]:
        """Validate JSON against schema and return any validation errors."""
        if schema_type not in self.schemas:
            return ["Unknown schema type"]
            
        try:
            validate(instance=data, schema=self.schemas[schema_type])
            return []
        except ValidationError as e:
            return [e.message]
            
    async def _analyze_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze JSON content for potential issues or insights."""
        # Convert data to string for the model
        json_str = json.dumps(data, indent=2)
        
        prompt = f"""Analyze the following JSON data for potential issues, risks, or notable patterns.
        Respond in JSON format with the following structure:
        {{
            "schema_type": "transaction|user_profile|unknown",
            "risk_level": "high|medium|low",
            "findings": ["finding1", "finding2"],
            "recommendations": ["rec1", "rec2"]
        }}
        
        JSON data:
        {json_str}
        """
        
        try:
            response = await self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            return {
                "schema_type": "unknown",
                "risk_level": "unknown",
                "findings": [],
                "recommendations": []
            }
            
    async def process(self, file_path: Path) -> Dict[str, Any]:
        """Process JSON file and return analysis."""
        try:
            # Read and parse JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Analyze content
            analysis = await self._analyze_content(data)
            
            # Validate against schema if type is known
            validation_errors = []
            if analysis["schema_type"] in self.schemas:
                validation_errors = self._validate_schema(data, analysis["schema_type"])
                
            # Check for high-value transactions
            high_value_threshold = 10000  # Configure as needed
            is_high_value = False
            if analysis["schema_type"] == "transaction":
                amount = data.get("amount", 0)
                is_high_value = amount >= high_value_threshold
                
            # Prepare result
            result = {
                "content": {
                    "schema_type": analysis["schema_type"],
                    "validation_errors": validation_errors,
                    "data_summary": {
                        k: v for k, v in data.items()
                        if k in ["id", "timestamp", "type", "status"]  # Safe fields to include
                    }
                },
                "analysis": analysis,
                "flags": {
                    "is_valid": len(validation_errors) == 0,
                    "is_high_value": is_high_value,
                    "needs_review": analysis["risk_level"] == "high" or is_high_value
                },
                "suggested_actions": []
            }
            
            # Add suggested actions based on analysis
            if is_high_value:
                result["suggested_actions"].append({
                    "type": "risk_alert",
                    "priority": "high",
                    "reason": f"High-value transaction (Amount: {data.get('amount')})"
                })
                
            if analysis["risk_level"] == "high":
                result["suggested_actions"].append({
                    "type": "flag_for_review",
                    "priority": "high",
                    "reason": "High risk level detected"
                })
                
            if not result["flags"]["is_valid"]:
                result["suggested_actions"].append({
                    "type": "validation_error",
                    "priority": "medium",
                    "reason": "Schema validation failed"
                })
                
            logger.info(f"JSON processing complete for {file_path.name}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"JSON processing error: {str(e)}")
            raise 