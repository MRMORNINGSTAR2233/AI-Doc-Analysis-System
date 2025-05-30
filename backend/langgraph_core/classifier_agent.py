import os
import google.generativeai as genai
from pathlib import Path
import json
import logging
from typing import Dict, Any
import mimetypes
from email import message_from_file
import pypdf

logger = logging.getLogger(__name__)

class ClassifierAgent:
    def __init__(self):
        # Initialize Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Define business intents
        self.business_intents = {
            "COMPLAINT": ["dissatisfied", "angry", "issue", "problem", "refund"],
            "INVOICE": ["invoice", "payment", "amount", "due", "bill"],
            "RFQ": ["quote", "proposal", "pricing", "estimate", "cost"],
            "REGULATION": ["compliance", "GDPR", "FDA", "regulation", "policy"],
            "FRAUD_RISK": ["suspicious", "fraud", "unauthorized", "alert", "risk"]
        }
        
    async def _detect_format(self, file_path: Path) -> str:
        """Detect file format based on content and extension."""
        mime_type = mimetypes.guess_type(file_path)[0]
        
        # Try extension-based detection first
        if file_path.suffix.lower() in ['.pdf']:
            return "pdf"
        elif file_path.suffix.lower() in ['.json']:
            return "json"
        elif file_path.suffix.lower() in ['.eml', '.msg']:
            return "email"
            
        # Try content-based detection
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                # Try parsing as JSON
                json.loads(content)
                return "json"
        except:
            pass
            
        try:
            # Try parsing as email
            with open(file_path, 'r') as f:
                msg = message_from_file(f)
                if msg.get('subject') or msg.get('from'):
                    return "email"
        except:
            pass
            
        try:
            # Try parsing as PDF
            pypdf.PdfReader(file_path)
            return "pdf"
        except:
            pass
            
        return "unknown"
        
    async def _detect_intent(self, content: str) -> Dict[str, Any]:
        """Detect business intent using Gemini."""
        prompt = f"""Analyze this business document content and determine its intent.
        Focus on key indicators like:
        - Complaints or dissatisfaction
        - Financial transactions or invoices
        - Requests for quotes/proposals
        - Regulatory/compliance matters
        - Risk/fraud indicators

        Provide your analysis in this exact JSON format:
        {{
            "primary_intent": "COMPLAINT|INVOICE|RFQ|REGULATION|FRAUD_RISK",
            "confidence": 0.0-1.0,
            "keywords_found": ["key1", "key2"],
            "reasoning": "Brief explanation"
        }}

        Content to analyze:
        ---
        {content[:2000]}
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
            
            # Validate intent against known types
            valid_intents = ["COMPLAINT", "INVOICE", "RFQ", "REGULATION", "FRAUD_RISK"]
            if result["primary_intent"] not in valid_intents:
                logger.warning(f"Invalid intent detected: {result['primary_intent']}")
                result["primary_intent"] = "UNKNOWN"
                result["confidence"] = 0.0
            
            # Ensure confidence is valid
            result["confidence"] = float(result["confidence"])
            if not 0 <= result["confidence"] <= 1:
                result["confidence"] = max(0.0, min(1.0, result["confidence"]))
            
            # Ensure keywords is a list
            if not isinstance(result["keywords_found"], list):
                result["keywords_found"] = []
            
            return result
            
        except Exception as e:
            logger.error(f"Intent detection failed: {str(e)}")
            return {
                "primary_intent": "UNKNOWN",
                "confidence": 0.0,
                "keywords_found": [],
                "reasoning": "Failed to analyze document content"
            }
            
    async def _extract_preview(self, file_path: Path, format: str) -> str:
        """Extract content preview for intent analysis."""
        try:
            if format == "pdf":
                reader = pypdf.PdfReader(file_path)
                return " ".join([page.extract_text() for page in reader.pages])[:2000]
            elif format == "json":
                with open(file_path, 'r') as f:
                    return json.dumps(json.load(f), indent=2)
            elif format == "email":
                with open(file_path, 'r') as f:
                    msg = message_from_file(f)
                    subject = msg.get('subject', '')
                    body = msg.get_payload()
                    return f"Subject: {subject}\n\nBody: {body}"
            else:
                with open(file_path, 'r') as f:
                    return f.read()[:2000]
        except Exception as e:
            logger.error(f"Content extraction failed: {str(e)}")
            return ""
            
    async def classify(self, file_path: Path) -> Dict[str, Any]:
        """Classify document format and intent."""
        try:
            # Detect format
            format = await self._detect_format(file_path)
            
            # Extract content preview
            content_preview = await self._extract_preview(file_path, format)
            
            # Detect intent
            intent_analysis = await self._detect_intent(content_preview)
            
            # Prepare classification result
            result = {
                "format": format,
                "intent": {
                    "type": intent_analysis["primary_intent"],
                    "confidence": intent_analysis["confidence"],
                    "keywords": intent_analysis["keywords_found"],
                    "reasoning": intent_analysis["reasoning"]
                },
                "metadata": {
                    "filename": file_path.name,
                    "size": os.path.getsize(file_path),
                    "timestamp": os.path.getctime(file_path)
                },
                "routing": {
                    "requires_immediate_attention": 
                        intent_analysis["primary_intent"] in ["COMPLAINT", "FRAUD_RISK"] 
                        and intent_analysis["confidence"] > 0.7,
                    "suggested_priority": "high" if intent_analysis["confidence"] > 0.8 else "medium"
                }
            }
            
            logger.info(f"Classification complete for {file_path.name}: {result['format']} - {result['intent']['type']}")
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            raise 