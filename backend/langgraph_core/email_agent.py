import google.generativeai as genai
from pathlib import Path
import os
from typing import Dict, Any
from email import message_from_file
from bs4 import BeautifulSoup
import re

class EmailAgent:
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        
        self.tone_examples = """
        Example 1:
        Email: "I am writing to express my extreme frustration with your service. This is unacceptable and I demand immediate action!"
        Tone: angry
        
        Example 2:
        Email: "I would greatly appreciate your assistance with this matter. Thank you for your time."
        Tone: polite
        
        Example 3:
        Email: "If this issue is not resolved within 24 hours, I will be forced to escalate this to your superiors and take legal action."
        Tone: threatening
        
        Example 4:
        Email: "This is my third attempt to contact you. Please escalate this to your supervisor immediately."
        Tone: escalation
        """
        
    def _extract_html_content(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
        
    def _extract_urgency(self, content: str) -> str:
        urgency_indicators = {
            'high': [
                r'urgent',
                r'asap',
                r'emergency',
                r'immediate',
                r'critical',
                r'deadline',
                r'priority'
            ],
            'medium': [
                r'important',
                r'attention',
                r'please respond',
                r'follow[- ]?up',
                r'review'
            ],
            'low': [
                r'fyi',
                r'update',
                r'information',
                r'newsletter'
            ]
        }
        
        content = content.lower()
        
        for level, patterns in urgency_indicators.items():
            if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                return level
                
        return 'normal'
        
    async def _detect_tone(self, content: str) -> str:
        prompt = f"""
        Based on the following examples, classify the tone of the given email:
        
        {self.tone_examples}
        
        Email: {content}
        Tone:"""
        
        response = self.model.generate_content(prompt)
        return response.text.strip()
        
    async def _extract_request(self, content: str) -> Dict[str, Any]:
        prompt = """
        Extract the main request or issue from the following email. Format the response as JSON with these fields:
        - main_topic: The primary subject or concern
        - action_needed: What needs to be done
        - timeline: Any mentioned deadlines or timeframes (or "none" if not specified)
        
        Email:
        {content}
        """
        
        response = self.model.generate_content(prompt.format(content=content))
        try:
            return json.loads(response.text)
        except:
            return {
                "main_topic": "unknown",
                "action_needed": "unknown",
                "timeline": "none"
            }
            
    async def process(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            email = message_from_file(f)
            
        # Extract content
        content = ""
        if email.is_multipart():
            for part in email.walk():
                if part.get_content_type() == "text/plain":
                    content += part.get_payload()
                elif part.get_content_type() == "text/html":
                    content += self._extract_html_content(part.get_payload())
        else:
            content = email.get_payload()
            
        # Process email
        tone = await self._detect_tone(content)
        urgency = self._extract_urgency(content)
        request_info = await self._extract_request(content)
        
        result = {
            "sender": email.get("from"),
            "subject": email.get("subject"),
            "date": email.get("date"),
            "tone": tone,
            "urgency": urgency,
            "request_info": request_info,
            "needs_escalation": tone in ["angry", "threatening", "escalation"] and urgency == "high"
        }
        
        return result 