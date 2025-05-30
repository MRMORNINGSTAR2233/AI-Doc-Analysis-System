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
            "date": msg.get("date", ""),
            "cc": msg.get("cc", ""),
            "reply-to": msg.get("reply-to", "")
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
        
    async def _analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze email content for sentiment, urgency, and tone."""
        prompt = f"""Analyze the following email for sentiment, urgency, and tone.
        Respond in JSON format with:
        {{
            "sentiment": {{
                "type": "positive|negative|neutral",
                "score": 0.0-1.0,
                "key_phrases": ["phrase1", "phrase2"]
            }},
            "urgency": {{
                "level": "high|medium|low",
                "time_sensitive": true|false,
                "deadline_mentioned": true|false,
                "deadline_text": "extracted deadline if mentioned"
            }},
            "tone": {{
                "primary": "angry|polite|neutral|demanding|appreciative",
                "formality": "formal|informal",
                "politeness_score": 0.0-1.0
            }},
            "issue": {{
                "category": "technical|billing|service|support|other",
                "summary": "brief description of the main issue/request",
                "action_items": ["action1", "action2"]
            }}
        }}
        
        Email text:
        {text}
        """
        
        try:
            response = await self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            return {
                "sentiment": {"type": "unknown", "score": 0.0, "key_phrases": []},
                "urgency": {"level": "unknown", "time_sensitive": False, "deadline_mentioned": False},
                "tone": {"primary": "unknown", "formality": "unknown", "politeness_score": 0.0},
                "issue": {"category": "unknown", "summary": "", "action_items": []}
            }
            
    async def process(self, file_path: Path) -> Dict[str, Any]:
        """Process email file and return analysis."""
        try:
            # Extract email content
            email_content = self._extract_email_content(file_path)
            
            # Analyze content
            analysis = await self._analyze_content(
                f"Subject: {email_content['headers']['subject']}\n\n{email_content['body']}"
            )
            
            # Prepare result
            result = {
                "metadata": {
                    "sender": email_content["headers"]["from"],
                    "recipient": email_content["headers"]["to"],
                    "cc": email_content["headers"]["cc"],
                    "reply_to": email_content["headers"]["reply-to"],
                    "date": email_content["headers"]["date"],
                    "subject": email_content["headers"]["subject"]
                },
                "content": {
                    "body_preview": email_content["body"][:1000] + "..." if len(email_content["body"]) > 1000 else email_content["body"]
                },
                "analysis": analysis,
                "suggested_actions": []
            }
            
            # Determine actions based on analysis
            if analysis["sentiment"]["type"] == "negative" and analysis["urgency"]["level"] == "high":
                result["suggested_actions"].append({
                    "type": "escalate",
                    "priority": "high",
                    "reason": "Negative sentiment with high urgency"
                })
                
            if analysis["tone"]["primary"] == "angry" and analysis["tone"]["politeness_score"] < 0.3:
                result["suggested_actions"].append({
                    "type": "flag_for_review",
                    "priority": "high",
                    "reason": "Angry tone detected"
                })
                
            if analysis["urgency"]["time_sensitive"]:
                result["suggested_actions"].append({
                    "type": "set_deadline",
                    "priority": "high",
                    "deadline": analysis["urgency"]["deadline_text"],
                    "reason": "Time-sensitive request"
                })
                
            # Add memory trace
            result["memory_trace"] = {
                "processing_steps": [
                    {"step": "content_extraction", "status": "success"},
                    {"step": "sentiment_analysis", "status": "success", "score": analysis["sentiment"]["score"]},
                    {"step": "urgency_detection", "status": "success", "level": analysis["urgency"]["level"]},
                    {"step": "tone_analysis", "status": "success", "tone": analysis["tone"]["primary"]}
                ],
                "decision_factors": {
                    "escalation_reason": "negative_sentiment" if analysis["sentiment"]["type"] == "negative" else None,
                    "urgency_reason": analysis["urgency"]["level"] if analysis["urgency"]["level"] == "high" else None,
                    "tone_reason": analysis["tone"]["primary"] if analysis["tone"]["primary"] == "angry" else None
                }
            }
            
            logger.info(f"Email processing complete for {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Email processing error: {str(e)}")
            raise 