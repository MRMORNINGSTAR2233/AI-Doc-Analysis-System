import google.generativeai as genai
from pathlib import Path
import json
import os
from typing import Dict, Any
import mimetypes
from email import message_from_file
import pypdf

class ClassifierAgent:
    def __init__(self):
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Few-shot examples for intent classification
        self.few_shot_examples = """
        Example 1:
        Content: "Please provide a quote for 1000 units of product XYZ..."
        Intent: RFQ (Request for Quote)
        
        Example 2:
        Content: "I am extremely dissatisfied with the service..."
        Intent: Complaint
        
        Example 3:
        Content: "Invoice #12345 for services rendered..."
        Intent: Invoice
        
        Example 4:
        Content: "New GDPR compliance requirements..."
        Intent: Regulation
        
        Example 5:
        Content: "Suspicious transaction pattern detected..."
        Intent: Fraud Risk
        """
        
    async def _detect_format(self, file_path: Path) -> str:
        mime_type = mimetypes.guess_type(file_path)[0]
        
        if mime_type == "application/pdf":
            return "pdf"
        elif mime_type in ["application/json", "text/json"]:
            return "json"
        elif mime_type in ["message/rfc822", "text/plain"]:
            try:
                with open(file_path, 'r') as f:
                    message_from_file(f)
                return "email"
            except:
                pass
        
        # If mime type detection fails, try content-based detection
        try:
            with open(file_path, 'r') as f:
                json.load(f)
                return "json"
        except:
            pass
            
        try:
            pypdf.PdfReader(file_path)
            return "pdf"
        except:
            pass
            
        return "unknown"
        
    async def _detect_intent(self, content: str) -> str:
        prompt = f"""
        Based on the following few-shot examples, classify the intent of the given content:
        
        {self.few_shot_examples}
        
        Content: {content}
        Intent:"""
        
        response = self.model.generate_content(prompt)
        return response.text.strip()
        
    async def _extract_metadata(self, file_path: Path, format: str) -> Dict[str, Any]:
        metadata = {
            "filename": file_path.name,
            "size": os.path.getsize(file_path),
            "created": os.path.getctime(file_path)
        }
        
        if format == "pdf":
            reader = pypdf.PdfReader(file_path)
            metadata.update({
                "pages": len(reader.pages),
                "pdf_info": reader.metadata
            })
        elif format == "email":
            with open(file_path, 'r') as f:
                email = message_from_file(f)
                metadata.update({
                    "subject": email.get("subject"),
                    "from": email.get("from"),
                    "date": email.get("date")
                })
                
        return metadata
        
    async def classify(self, file_path: Path) -> Dict[str, Any]:
        format = await self._detect_format(file_path)
        
        # Read content for intent detection
        content = ""
        if format == "pdf":
            reader = pypdf.PdfReader(file_path)
            content = " ".join([page.extract_text() for page in reader.pages])
        elif format == "json":
            with open(file_path, 'r') as f:
                content = f.read()
        elif format == "email":
            with open(file_path, 'r') as f:
                email = message_from_file(f)
                content = email.get_payload()
                
        intent = await self._detect_intent(content)
        metadata = await self._extract_metadata(file_path, format)
        
        return {
            "format": format,
            "intent": intent,
            "metadata": metadata,
            "file_path": str(file_path)
        } 