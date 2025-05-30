from pathlib import Path
import json
from typing import Dict, Any
from jsonschema import validate, ValidationError

class JSONAgent:
    def __init__(self):
        # Define base schema for different types of JSON inputs
        self.schemas = {
            "webhook": {
                "type": "object",
                "required": ["event_type", "timestamp", "data"],
                "properties": {
                    "event_type": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "data": {"type": "object"}
                }
            },
            "transaction": {
                "type": "object",
                "required": ["transaction_id", "amount", "currency", "customer"],
                "properties": {
                    "transaction_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "currency": {"type": "string"},
                    "customer": {
                        "type": "object",
                        "required": ["id", "name"],
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "email": {"type": "string"}
                        }
                    }
                }
            }
        }
        
    def _detect_schema_type(self, data: Dict) -> str:
        # Try to determine which schema the JSON matches
        if all(key in data for key in ["event_type", "timestamp", "data"]):
            return "webhook"
        elif all(key in data for key in ["transaction_id", "amount", "currency"]):
            return "transaction"
        else:
            return "unknown"
            
    def _validate_schema(self, data: Dict, schema_type: str) -> Dict[str, Any]:
        if schema_type not in self.schemas:
            return {
                "valid": False,
                "errors": ["Unknown schema type"]
            }
            
        try:
            validate(instance=data, schema=self.schemas[schema_type])
            return {
                "valid": True,
                "errors": []
            }
        except ValidationError as e:
            return {
                "valid": False,
                "errors": [str(e)]
            }
            
    def _check_risk_indicators(self, data: Dict) -> Dict[str, Any]:
        risk_indicators = []
        
        # Check for transaction-specific risks
        if "amount" in data:
            if data["amount"] > 10000:
                risk_indicators.append("High value transaction")
                
        # Check for suspicious patterns
        if "event_type" in data and data["event_type"] in ["account.created", "password.changed"]:
            if "ip_address" in data.get("data", {}):
                risk_indicators.append("Security-sensitive operation")
                
        return {
            "has_risks": len(risk_indicators) > 0,
            "risk_indicators": risk_indicators
        }
        
    async def process(self, file_path: Path) -> Dict[str, Any]:
        # Read and parse JSON
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                return {
                    "valid": False,
                    "schema_type": "invalid",
                    "validation_result": {
                        "valid": False,
                        "errors": [f"Invalid JSON: {str(e)}"]
                    },
                    "risk_assessment": {
                        "has_risks": False,
                        "risk_indicators": []
                    }
                }
                
        # Process JSON
        schema_type = self._detect_schema_type(data)
        validation_result = self._validate_schema(data, schema_type)
        risk_assessment = self._check_risk_indicators(data)
        
        return {
            "valid": validation_result["valid"],
            "schema_type": schema_type,
            "validation_result": validation_result,
            "risk_assessment": risk_assessment,
            "data": data
        } 