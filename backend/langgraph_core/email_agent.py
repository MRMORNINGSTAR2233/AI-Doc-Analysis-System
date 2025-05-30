import os
import google.generativeai as genai
from pathlib import Path
import json
import logging
from email import message_from_file
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class EmailAgent:
    def __init__(self):
        # Initialize Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def _extract_email_content(self, file_path: Path) -> Dict[str, str]:
        """Extract content from email file."""
        with open(file_path) as email_file:
            msg = message_from_file(email_file)
            
        # Get basic headers
        headers = {
            "subject": msg.get("subject", ""),
            "from": msg.get("from", ""),
            "to": msg.get("to", ""),
            "date": msg.get("date", "")
        }
        
        # Get body content
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
                elif part.get_content_type() == "text/html":
                    html = part.get_payload(decode=True).decode()
                    soup = BeautifulSoup(html, 'html.parser')
                    body = soup.get_text()
                    break
        else:
            body = msg.get_payload(decode=True).decode()
            
        return {
            "headers": headers,
            "body": body
        }
        
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze email sentiment and urgency."""
        prompt = f"""Analyze the following email text for sentiment and urgency.
        Respond in JSON format with the following structure:
        {{
            "sentiment": "positive|negative|neutral",
            "sentiment_score": 0.0-1.0,
            "urgency": "high|medium|low",
            "key_topics": ["topic1", "topic2"],
            "requires_immediate_action": true|false
        }}
        
        Email text:
        {text}
        """
        
        try:
            response = await self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {
                "sentiment": "unknown",
                "sentiment_score": 0.0,
                "urgency": "unknown",
                "key_topics": [],
                "requires_immediate_action": False
            }
            
    async def process(self, file_path: Path) -> Dict[str, Any]:
        """Process email file and return analysis."""
        try:
            # Extract email content
            email_content = self._extract_email_content(file_path)
            
            # Analyze sentiment
            sentiment_analysis = await self._analyze_sentiment(
                f"Subject: {email_content['headers']['subject']}\n\n{email_content['body']}"
            )
            
            # Prepare result
            result = {
                "metadata": email_content["headers"],
                "content": {
                    "body": email_content["body"][:1000] + "..." if len(email_content["body"]) > 1000 else email_content["body"]
                },
                "analysis": sentiment_analysis,
                "suggested_actions": []
            }
            
            # Determine suggested actions based on analysis
            if sentiment_analysis["sentiment"] == "negative" and sentiment_analysis["urgency"] == "high":
                result["suggested_actions"].append({
                    "type": "escalate",
                    "priority": "high",
                    "reason": "Negative sentiment with high urgency"
                })
                
            if sentiment_analysis["requires_immediate_action"]:
                result["suggested_actions"].append({
                    "type": "flag_for_review",
                    "priority": "high",
                    "reason": "Requires immediate attention"
                })
                
            logger.info(f"Email processing complete for {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Email processing error: {str(e)}")
            raise 