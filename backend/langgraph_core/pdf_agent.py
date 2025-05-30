import os
import google.generativeai as genai
from pathlib import Path
import json
import logging
from typing import Dict, Any, List
import pypdf

logger = logging.getLogger(__name__)

class PDFAgent:
    def __init__(self):
        # Initialize Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def _extract_text(self, file_path: Path) -> str:
        """Extract text content from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf = pypdf.PdfReader(file)
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"PDF text extraction failed: {str(e)}")
            raise
            
    async def _analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze PDF content for document type, compliance, and key information."""
        prompt = f"""Analyze the following document text for type, compliance requirements, and key information.
        Respond in JSON format with the following structure:
        {{
            "document_type": "invoice|contract|report|other",
            "compliance_requirements": ["requirement1", "requirement2"],
            "sensitive_information": {{
                "contains_pii": true|false,
                "contains_financial": true|false,
                "contains_medical": true|false
            }},
            "key_findings": ["finding1", "finding2"],
            "risk_level": "high|medium|low"
        }}
        
        Document text:
        {text[:2000]}...  # Limit text length for API
        """
        
        try:
            response = await self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            return {
                "document_type": "unknown",
                "compliance_requirements": [],
                "sensitive_information": {
                    "contains_pii": False,
                    "contains_financial": False,
                    "contains_medical": False
                },
                "key_findings": [],
                "risk_level": "unknown"
            }
            
    def _check_compliance(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for compliance issues based on document analysis."""
        compliance_issues = []
        
        # Check for PII handling requirements
        if analysis["sensitive_information"]["contains_pii"]:
            compliance_issues.append({
                "type": "pii_handling",
                "description": "Document contains PII - ensure GDPR/CCPA compliance",
                "severity": "high"
            })
            
        # Check for financial data requirements
        if analysis["sensitive_information"]["contains_financial"]:
            compliance_issues.append({
                "type": "financial_compliance",
                "description": "Document contains financial data - ensure SOX compliance",
                "severity": "high"
            })
            
        # Check for medical data requirements
        if analysis["sensitive_information"]["contains_medical"]:
            compliance_issues.append({
                "type": "medical_compliance",
                "description": "Document contains medical information - ensure HIPAA compliance",
                "severity": "high"
            })
            
        return compliance_issues
        
    async def process(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF file and return analysis."""
        try:
            # Extract text
            text = self._extract_text(file_path)
            
            # Analyze content
            analysis = await self._analyze_content(text)
            
            # Check compliance
            compliance_issues = self._check_compliance(analysis)
            
            # Prepare result
            result = {
                "metadata": {
                    "file_name": file_path.name,
                    "document_type": analysis["document_type"]
                },
                "content": {
                    "text_preview": text[:1000] + "..." if len(text) > 1000 else text,
                    "page_count": len(pypdf.PdfReader(file_path).pages)
                },
                "analysis": {
                    "compliance_requirements": analysis["compliance_requirements"],
                    "sensitive_information": analysis["sensitive_information"],
                    "key_findings": analysis["key_findings"],
                    "risk_level": analysis["risk_level"],
                    "compliance_issues": compliance_issues
                },
                "suggested_actions": []
            }
            
            # Add suggested actions based on analysis
            if analysis["risk_level"] == "high":
                result["suggested_actions"].append({
                    "type": "risk_alert",
                    "priority": "high",
                    "reason": "High risk level detected in document"
                })
                
            if compliance_issues:
                result["suggested_actions"].append({
                    "type": "compliance_review",
                    "priority": "high",
                    "reason": f"Found {len(compliance_issues)} compliance issues"
                })
                
            if any(analysis["sensitive_information"].values()):
                result["suggested_actions"].append({
                    "type": "data_protection",
                    "priority": "high",
                    "reason": "Document contains sensitive information"
                })
                
            logger.info(f"PDF processing complete for {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            raise 