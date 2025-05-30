import os
import google.generativeai as genai
from pathlib import Path
import json
import logging
from typing import Dict, Any, List
from jsonschema import validate, ValidationError, Draft7Validator

logger = logging.getLogger(__name__)

class JSONAgent:
    def __init__(self):
        # Initialize Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Define common JSON schemas
        self.schemas = {
            "transaction": {
                "type": "object",
                "required": ["amount", "currency", "timestamp", "description"],
                "properties": {
                    "amount": {"type": "number", "minimum": 0},
                    "currency": {"type": "string", "minLength": 3, "maxLength": 3},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "description": {"type": "string"},
                    "customer": {
                        "type": "object",
                        "required": ["id", "name"],
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "email": {"type": "string", "format": "email"}
                        }
                    }
                }
            },
            "user_profile": {
                "type": "object",
                "required": ["id", "name", "email"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                    "preferences": {"type": "object"}
                }
            }
        }
        
    def _validate_schema(self, data: Dict[str, Any], schema_type: str) -> List[str]:
        """Validate JSON against schema and return validation errors."""
        if schema_type not in self.schemas:
            return ["Unknown schema type"]
            
        validator = Draft7Validator(self.schemas[schema_type])
        return [error.message for error in validator.iter_errors(data)]
        
    def _detect_missing_required(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Detect missing required fields."""
        missing = []
        if "required" in schema:
            for field in schema["required"]:
                if field not in data:
                    missing.append(field)
                    
        if "properties" in schema:
            for field, value in data.items():
                if field in schema["properties"] and isinstance(value, dict):
                    missing.extend(self._detect_missing_required(value, schema["properties"][field]))
                    
        return missing
        
    def _detect_type_mismatches(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Detect type mismatches in the data."""
        mismatches = []
        if "properties" in schema:
            for field, value in data.items():
                if field in schema["properties"]:
                    expected_type = schema["properties"][field]["type"]
                    if expected_type == "number" and not isinstance(value, (int, float)):
                        mismatches.append(f"Field '{field}' should be a number")
                    elif expected_type == "string" and not isinstance(value, str):
                        mismatches.append(f"Field '{field}' should be a string")
                    elif expected_type == "object" and not isinstance(value, dict):
                        mismatches.append(f"Field '{field}' should be an object")
                        
        return mismatches
        
    async def _analyze_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transaction risk using Gemini."""
        prompt = f"""Analyze this financial transaction data for risks and anomalies.
        Focus on:
        - Transaction amount and currency
        - Customer profile and history
        - Payment method and timing
        - Unusual patterns or red flags

        Provide your analysis in this exact JSON format:
        {{
            "risk_level": "high|medium|low",
            "risk_factors": ["factor1", "factor2"],
            "anomalies_detected": ["anomaly1", "anomaly2"],
            "recommendations": ["action1", "action2"],
            "requires_review": true|false
        }}

        Transaction data:
        ---
        {json.dumps(data, indent=2)}
        ---

        Respond with ONLY the JSON:"""

        try:
            # Create new model instance for clean context
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            
            # Parse and validate response
            response_text = response.text.strip()
            if "```" in response_text:
                response_text = response_text.split("```")[1].replace("json", "").strip()
            
            result = json.loads(response_text)
            
            # Validate risk level
            valid_risk_levels = ["high", "medium", "low"]
            if result["risk_level"] not in valid_risk_levels:
                result["risk_level"] = "unknown"
            
            # Ensure all lists are valid
            for field in ["risk_factors", "anomalies_detected", "recommendations"]:
                if not isinstance(result[field], list):
                    result[field] = []
            
            # Ensure requires_review is boolean
            result["requires_review"] = bool(result.get("requires_review", True))
            
            return result
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {str(e)}")
            return {
                "risk_level": "unknown",
                "risk_factors": [],
                "anomalies_detected": [],
                "recommendations": [],
                "requires_review": True
            }
            
    async def process(self, file_path: Path) -> Dict[str, Any]:
        """Process JSON file and return analysis."""
        try:
            # Read and parse JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Determine schema type
            schema_type = "transaction" if "amount" in data else "user_profile"
            
            # Validate schema
            validation_errors = self._validate_schema(data, schema_type)
            missing_fields = self._detect_missing_required(data, self.schemas[schema_type])
            type_mismatches = self._detect_type_mismatches(data, self.schemas[schema_type])
            
            # Check for high-value transactions
            is_high_value = False
            if schema_type == "transaction" and "amount" in data:
                is_high_value = float(data["amount"]) >= 10000
                
            # Analyze risk
            risk_analysis = await self._analyze_risk(data)
            
            # Prepare result
            result = {
                "validation": {
                    "schema_type": schema_type,
                    "is_valid": len(validation_errors) == 0,
                    "errors": validation_errors,
                    "missing_fields": missing_fields,
                    "type_mismatches": type_mismatches
                },
                "content": {
                    "data_preview": {k: v for k, v in data.items() if k not in ["sensitive_data", "credentials"]}
                },
                "risk_assessment": {
                    "is_high_value": is_high_value,
                    "risk_level": risk_analysis["risk_level"],
                    "risk_factors": risk_analysis["risk_factors"],
                    "anomalies": risk_analysis["anomalies_detected"],
                    "recommendations": risk_analysis["recommendations"]
                },
                "suggested_actions": []
            }
            
            # Add suggested actions
            if is_high_value:
                result["suggested_actions"].append({
                    "type": "risk_alert",
                    "priority": "high",
                    "reason": f"High-value transaction: {data.get('amount')} {data.get('currency', 'USD')}"
                })
                
            if risk_analysis["risk_level"] == "high":
                result["suggested_actions"].append({
                    "type": "flag_for_review",
                    "priority": "high",
                    "reason": "High risk level detected"
                })
                
            if not result["validation"]["is_valid"]:
                result["suggested_actions"].append({
                    "type": "validation_error",
                    "priority": "medium",
                    "reason": "Schema validation failed"
                })
                
            # Add memory trace
            result["memory_trace"] = {
                "processing_steps": [
                    {"step": "schema_validation", "status": "success" if result["validation"]["is_valid"] else "failed"},
                    {"step": "risk_analysis", "status": "success", "risk_level": risk_analysis["risk_level"]},
                    {"step": "high_value_check", "status": "success", "is_high_value": is_high_value}
                ],
                "decision_factors": {
                    "validation_errors": len(validation_errors) > 0,
                    "high_value": is_high_value,
                    "high_risk": risk_analysis["risk_level"] == "high"
                }
            }
            
            logger.info(f"JSON processing complete for {file_path.name}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"JSON processing error: {str(e)}")
            raise 