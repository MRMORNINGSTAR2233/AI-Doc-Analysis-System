import os
import google.generativeai as genai
from pathlib import Path
import json
import logging
from typing import Dict, Any, List
from jsonschema import validate, ValidationError, Draft7Validator
import asyncio

logger = logging.getLogger(__name__)

class JSONAgent:
    def __init__(self):
        # Initialize Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key or api_key == "your_google_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required. "
                "Please set a valid Google Generative AI API key in your .env file."
            )
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("JSON Agent: Successfully initialized Google Generative AI model")
        except Exception as e:
            logger.error(f"JSON Agent: Failed to initialize Gemini model: {str(e)}")
            raise RuntimeError(f"Could not initialize Google Generative AI: {str(e)}")
        
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
        prompt = f"""Analyze this financial transaction or user data for risks, patterns, and business impact.

        Focus on these key areas:
        - Transaction amount and patterns
        - Customer profile and history
        - Risk indicators and anomalies
        - Compliance requirements
        - Business impact and urgency

        You MUST respond with ONLY a JSON object in this exact format:
        {{
            "risk_level": "high|medium|low",
            "risk_score": <float between 0.0-1.0>,
            "risk_factors": ["specific risk factor 1", "specific risk factor 2"],
            "anomalies_detected": ["specific anomaly 1", "specific anomaly 2"],
            "compliance_flags": ["specific compliance issue 1", "specific compliance issue 2"],
            "business_impact": {{
                "severity": "high|medium|low",
                "potential_loss": "<estimated value if applicable>",
                "urgency": "immediate|high|medium|low"
            }},
            "recommendations": ["specific action 1", "specific action 2"],
            "confidence_level": <float between 0.0-1.0>,
            "analysis_summary": "Brief explanation of key findings"
        }}

        Data to analyze:
        ---
        {json.dumps(data, indent=2)}
        ---"""

        try:
            # Create new model instance for clean context
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Generate response with safety settings
            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.3,
                    'top_p': 0.8,
                    'top_k': 40
                }
            )
            
            # Clean and parse response
            response_text = response.text.strip()
            if "```" in response_text:
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()
            
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response as JSON: {response_text}")
                return self._generate_fallback_risk_analysis(data, str(e))
            
            # Validate and clean result
            result = self._validate_risk_analysis(result, data)
            
            return result
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {str(e)}")
            return self._generate_fallback_risk_analysis(data, str(e))

    def _validate_risk_analysis(self, result: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean risk analysis result."""
        # Ensure valid risk level
        valid_risk_levels = ["high", "medium", "low"]
        if not isinstance(result.get("risk_level"), str) or result["risk_level"].lower() not in valid_risk_levels:
            result["risk_level"] = self._determine_fallback_risk_level(data)
        
        # Ensure valid risk score
        try:
            result["risk_score"] = float(result.get("risk_score", 0.5))
            result["risk_score"] = max(0.0, min(1.0, result["risk_score"]))
        except (TypeError, ValueError):
            result["risk_score"] = 0.5
        
        # Ensure valid lists
        for key in ["risk_factors", "anomalies_detected", "compliance_flags", "recommendations"]:
            if not isinstance(result.get(key), list):
                result[key] = []
            result[key] = [str(item) for item in result[key] if item]
        
        # Validate business impact
        if not isinstance(result.get("business_impact"), dict):
            result["business_impact"] = {
                "severity": "medium",
                "potential_loss": "Unknown",
                "urgency": "medium"
            }
        
        # Ensure valid confidence level
        try:
            result["confidence_level"] = float(result.get("confidence_level", 0.5))
            result["confidence_level"] = max(0.0, min(1.0, result["confidence_level"]))
        except (TypeError, ValueError):
            result["confidence_level"] = 0.5
        
        # Add confidence explanation
        result["confidence_explanation"] = self._get_confidence_explanation(result["confidence_level"])
        
        return result

    def _determine_fallback_risk_level(self, data: Dict[str, Any]) -> str:
        """Determine risk level based on basic data analysis."""
        risk_level = "medium"  # Default risk level
        
        try:
            # Check for high-value transactions
            if "amount" in data and float(data["amount"]) > 10000:
                risk_level = "high"
            
            # Check for suspicious patterns
            suspicious_keys = ["risk", "fraud", "alert", "warning", "suspicious"]
            if any(key in str(data).lower() for key in suspicious_keys):
                risk_level = "high"
            
            # Check for compliance-related content
            compliance_keys = ["compliance", "regulation", "policy", "legal"]
            if any(key in str(data).lower() for key in compliance_keys):
                risk_level = "high"
                
        except:
            pass
            
        return risk_level

    def _get_confidence_explanation(self, confidence: float) -> str:
        """Generate an explanation for the confidence score."""
        if confidence >= 0.8:
            return "High confidence based on clear data patterns and strong indicators"
        elif confidence >= 0.6:
            return "Medium confidence with moderate risk indicators present"
        elif confidence >= 0.4:
            return "Low-medium confidence due to mixed or unclear indicators"
        else:
            return "Low confidence due to insufficient or ambiguous data"

    def _generate_fallback_risk_analysis(self, data: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Generate a fallback risk analysis when the main analysis fails."""
        risk_level = self._determine_fallback_risk_level(data)
        
        # Extract basic metrics
        metrics = self._extract_basic_metrics(data)
        
        return {
            "risk_level": risk_level,
            "risk_score": 0.5 if risk_level == "medium" else 0.8,
            "risk_factors": [
                f"Automated fallback analysis due to: {error_msg}",
                *metrics["risk_factors"]
            ],
            "anomalies_detected": metrics["anomalies"],
            "compliance_flags": metrics["compliance_flags"],
            "business_impact": {
                "severity": risk_level,
                "potential_loss": metrics["potential_loss"],
                "urgency": "high" if risk_level == "high" else "medium"
            },
            "recommendations": [
                "Conduct manual review due to analysis limitations",
                "Verify transaction details with additional sources",
                *metrics["recommendations"]
            ],
            "confidence_level": 0.4,
            "confidence_explanation": "Limited confidence due to fallback analysis",
            "analysis_summary": f"Basic risk assessment completed with limited analysis capabilities. {metrics['summary']}"
        }

    def _extract_basic_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic metrics from the data for fallback analysis."""
        metrics = {
            "risk_factors": [],
            "anomalies": [],
            "compliance_flags": [],
            "recommendations": [],
            "potential_loss": "Unknown",
            "summary": ""
        }
        
        try:
            # Check for monetary values
            if "amount" in data:
                amount = float(data["amount"])
                metrics["potential_loss"] = f"${amount:,.2f}"
                if amount > 10000:
                    metrics["risk_factors"].append("High-value transaction")
                    metrics["recommendations"].append("Require additional approval for high-value transaction")
            
            # Check for customer data
            if "customer" in data:
                metrics["risk_factors"].append("Customer data present - requires privacy consideration")
                metrics["compliance_flags"].append("Personal data handling requirements")
            
            # Check for timestamps
            if "timestamp" in data:
                metrics["summary"] = f"Transaction dated {data['timestamp']}"
            
        except Exception as e:
            logger.error(f"Error extracting basic metrics: {str(e)}")
        
        return metrics
        
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