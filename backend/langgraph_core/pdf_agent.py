import os
import google.generativeai as genai
from pathlib import Path
import json
import logging
from typing import Dict, Any, List
import pypdf
import re

logger = logging.getLogger(__name__)

class PDFAgent:
    def __init__(self):
        # Initialize Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Define compliance keywords
        self.compliance_keywords = {
            "GDPR": ["gdpr", "data protection", "privacy", "personal data", "data subject"],
            "HIPAA": ["hipaa", "health", "medical", "patient", "phi"],
            "FDA": ["fda", "drug", "medical device", "clinical", "pharmaceutical"],
            "SOX": ["sox", "sarbanes", "financial reporting", "audit", "internal control"],
            "PCI": ["pci", "payment card", "credit card", "cardholder", "merchant"]
        }
        
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
            
    def _extract_invoice_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract invoice line items using regex patterns."""
        items = []
        
        # Common invoice item patterns
        patterns = [
            r"\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD)?\s*[-–]\s*(.+?)(?=\$|\n|$)",
            r"(\d+)\s*x\s*(.+?)\s*@\s*\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)",
            r"(.+?)\s*\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD)?(?=\n|$)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) == 2:  # Amount and description
                    amount = float(match.group(1).replace(',', ''))
                    description = match.group(2).strip()
                    items.append({
                        "amount": amount,
                        "description": description
                    })
                elif len(match.groups()) == 3:  # Quantity, item, and unit price
                    quantity = int(match.group(1))
                    description = match.group(2).strip()
                    unit_price = float(match.group(3).replace(',', ''))
                    items.append({
                        "quantity": quantity,
                        "description": description,
                        "unit_price": unit_price,
                        "amount": quantity * unit_price
                    })
                    
        return items
        
    def _check_compliance_keywords(self, text: str) -> Dict[str, List[str]]:
        """Check for compliance-related keywords."""
        findings = {}
        text = text.lower()
        
        for regulation, keywords in self.compliance_keywords.items():
            matches = []
            for keyword in keywords:
                if keyword.lower() in text:
                    matches.append(keyword)
            if matches:
                findings[regulation] = matches
                
        return findings
        
    async def _analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze PDF content for document type, compliance, and key information."""
        prompt = f"""Analyze the following document text for type, compliance requirements, and key information.
        Respond in JSON format with:
        {{
            "document_type": "invoice|contract|report|policy|other",
            "compliance_requirements": ["requirement1", "requirement2"],
            "key_findings": {{
                "monetary_values": ["value1", "value2"],
                "dates": ["date1", "date2"],
                "entities": ["entity1", "entity2"],
                "critical_terms": ["term1", "term2"]
            }},
            "risk_assessment": {{
                "level": "high|medium|low",
                "factors": ["factor1", "factor2"],
                "requires_review": true|false
            }}
        }}
        
        Document text:
        {text[:3000]}...
        """
        
        try:
            response = await self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            return {
                "document_type": "unknown",
                "compliance_requirements": [],
                "key_findings": {
                    "monetary_values": [],
                    "dates": [],
                    "entities": [],
                    "critical_terms": []
                },
                "risk_assessment": {
                    "level": "unknown",
                    "factors": [],
                    "requires_review": True
                }
            }
            
    async def process(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF file and return analysis."""
        try:
            # Extract text
            text = self._extract_text(file_path)
            
            # Extract invoice items if present
            invoice_items = self._extract_invoice_items(text)
            
            # Check for compliance keywords
            compliance_findings = self._check_compliance_keywords(text)
            
            # Analyze content
            content_analysis = await self._analyze_content(text)
            
            # Calculate total value for invoices
            total_value = sum(item["amount"] for item in invoice_items) if invoice_items else 0
            
            # Prepare result
            result = {
                "metadata": {
                    "file_name": file_path.name,
                    "document_type": content_analysis["document_type"],
                    "page_count": len(pypdf.PdfReader(file_path).pages)
                },
                "content": {
                    "text_preview": text[:1000] + "..." if len(text) > 1000 else text,
                    "invoice_items": invoice_items if invoice_items else None,
                    "total_value": total_value if invoice_items else None
                },
                "analysis": {
                    "compliance": {
                        "regulations_mentioned": list(compliance_findings.keys()),
                        "keyword_matches": compliance_findings,
                        "requirements": content_analysis["compliance_requirements"]
                    },
                    "key_findings": content_analysis["key_findings"],
                    "risk_assessment": content_analysis["risk_assessment"]
                },
                "flags": {
                    "high_value": total_value >= 10000 if total_value else False,
                    "requires_review": content_analysis["risk_assessment"]["requires_review"],
                    "has_compliance_terms": len(compliance_findings) > 0
                },
                "suggested_actions": []
            }
            
            # Add suggested actions
            if result["flags"]["high_value"]:
                result["suggested_actions"].append({
                    "type": "high_value_alert",
                    "priority": "high",
                    "reason": f"Document contains high value amount: ${total_value:,.2f}"
                })
                
            if result["flags"]["has_compliance_terms"]:
                result["suggested_actions"].append({
                    "type": "compliance_review",
                    "priority": "high",
                    "reason": f"Found compliance terms: {', '.join(result['analysis']['compliance']['regulations_mentioned'])}"
                })
                
            if content_analysis["risk_assessment"]["level"] == "high":
                result["suggested_actions"].append({
                    "type": "risk_alert",
                    "priority": "high",
                    "reason": "High risk level detected in document"
                })
                
            # Add memory trace
            result["memory_trace"] = {
                "processing_steps": [
                    {"step": "text_extraction", "status": "success"},
                    {"step": "invoice_detection", "status": "success", "items_found": len(invoice_items) if invoice_items else 0},
                    {"step": "compliance_check", "status": "success", "regulations_found": len(compliance_findings)},
                    {"step": "content_analysis", "status": "success", "risk_level": content_analysis["risk_assessment"]["level"]}
                ],
                "decision_factors": {
                    "high_value": result["flags"]["high_value"],
                    "compliance_terms": result["flags"]["has_compliance_terms"],
                    "high_risk": content_analysis["risk_assessment"]["level"] == "high"
                }
            }
            
            logger.info(f"PDF processing complete for {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            raise 