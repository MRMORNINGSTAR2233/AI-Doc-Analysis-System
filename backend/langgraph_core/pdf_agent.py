import google.generativeai as genai
from pathlib import Path
import os
from typing import Dict, Any, List
import pypdf
import re

class PDFAgent:
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Compliance keywords
        self.compliance_keywords = {
            "GDPR": [
                r"GDPR",
                r"General Data Protection Regulation",
                r"data protection",
                r"personal data",
                r"data subject",
                r"data controller"
            ],
            "HIPAA": [
                r"HIPAA",
                r"Health Insurance Portability",
                r"protected health information",
                r"PHI",
                r"medical records"
            ],
            "FDA": [
                r"FDA",
                r"Food and Drug Administration",
                r"drug safety",
                r"clinical trials",
                r"medical device"
            ]
        }
        
    async def _extract_invoice_items(self, text: str) -> List[Dict[str, Any]]:
        prompt = """
        Extract line items from this invoice text. Format as JSON array with fields:
        - description: Item description
        - quantity: Numeric quantity
        - unit_price: Price per unit
        - total: Total price for line item
        
        Text:
        {text}
        """
        
        response = self.model.generate_content(prompt.format(text=text))
        try:
            items = json.loads(response.text)
            return items if isinstance(items, list) else []
        except:
            return []
            
    def _check_compliance_mentions(self, text: str) -> Dict[str, List[str]]:
        mentions = {}
        
        for regulation, patterns in self.compliance_keywords.items():
            matches = []
            for pattern in patterns:
                found = re.finditer(pattern, text, re.IGNORECASE)
                matches.extend([text[max(0, m.start()-50):m.end()+50].strip() for m in found])
            if matches:
                mentions[regulation] = matches
                
        return mentions
        
    def _extract_total_amount(self, text: str) -> float:
        # Common patterns for total amounts
        patterns = [
            r"total:?\s*[\$€£]?([\d,]+\.?\d*)",
            r"amount due:?\s*[\$€£]?([\d,]+\.?\d*)",
            r"grand total:?\s*[\$€£]?([\d,]+\.?\d*)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    amount_str = match.group(1).replace(',', '')
                    return float(amount_str)
                except:
                    continue
                    
        return 0.0
        
    async def process(self, file_path: Path) -> Dict[str, Any]:
        # Read PDF
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        # Process content
        invoice_items = await self._extract_invoice_items(text)
        total_amount = self._extract_total_amount(text)
        compliance_mentions = self._check_compliance_mentions(text)
        
        # Check for violations
        violations = []
        if total_amount > 10000:
            violations.append({
                "type": "high_value",
                "details": f"Invoice total (${total_amount:,.2f}) exceeds $10,000 threshold"
            })
            
        if compliance_mentions:
            violations.append({
                "type": "compliance",
                "details": "Document contains regulatory compliance terms",
                "mentions": compliance_mentions
            })
            
        return {
            "metadata": {
                "pages": len(reader.pages),
                "pdf_info": reader.metadata
            },
            "content": {
                "invoice_items": invoice_items,
                "total_amount": total_amount,
                "compliance_mentions": compliance_mentions
            },
            "violations": violations,
            "needs_risk_alert": len(violations) > 0
        } 