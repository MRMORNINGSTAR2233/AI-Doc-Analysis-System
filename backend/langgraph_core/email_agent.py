import os
import google.generativeai as genai
from pathlib import Path
import json
import logging
from typing import Dict, Any, List
from email import message_from_file, policy
from email.parser import Parser
from email.policy import default
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class EmailAgent:
    def __init__(self):
        # Initialize Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    async def process(self, file_path: Path) -> Dict[str, Any]:
        """Process email file and return analysis."""
        try:
            # Parse email
            email_data = self._parse_email(file_path)
            
            # Analyze content
            content_analysis = await self._analyze_content(email_data)
            
            # Extract entities
            entities = self._extract_entities(email_data["body"])
            
            # Generate actions
            actions = self._generate_actions(content_analysis, email_data, entities)
            
            # Prepare result
            result = {
                "metadata": {
                    "subject": email_data["subject"],
                    "from": email_data["from"],
                    "to": email_data["to"],
                    "date": email_data["date"],
                    "has_attachments": bool(email_data["attachments"])
                },
                "content": {
                    "body_preview": email_data["body"][:1000] + "..." if len(email_data["body"]) > 1000 else email_data["body"],
                    "attachments": email_data["attachments"]
                },
                "analysis": content_analysis,
                "entities": entities,
                "actions": actions
            }
            
            # Add memory trace
            result["memory_trace"] = {
                "processing_steps": [
                    {"step": "email_parsing", "status": "success"},
                    {"step": "content_analysis", "status": "success"},
                    {"step": "entity_extraction", "status": "success", "entities_found": sum(len(v) for v in entities.values())},
                    {"step": "action_generation", "status": "success", "actions_generated": len(actions)}
                ]
            }
            
            logger.info(f"Email processing complete for {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Email processing error: {str(e)}")
            raise
            
    def _parse_email(self, file_path: Path) -> Dict[str, Any]:
        """Parse email file and extract components."""
        try:
            with open(file_path, 'r') as f:
                msg = message_from_file(f, policy=policy.default)
                
            # Extract basic headers
            email_data = {
                "subject": msg.get("subject", ""),
                "from": msg.get("from", ""),
                "to": msg.get("to", ""),
                "date": msg.get("date", ""),
                "body": "",
                "attachments": []
            }
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        email_data["body"] += part.get_payload(decode=True).decode()
                    elif part.get_content_type() != "multipart/alternative":
                        email_data["attachments"].append({
                            "filename": part.get_filename(),
                            "type": part.get_content_type()
                        })
            else:
                email_data["body"] = msg.get_payload(decode=True).decode()
                
            return email_data
            
        except Exception as e:
            logger.error(f"Email parsing error: {str(e)}")
            raise
            
    async def _analyze_content(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze email content using Gemini."""
        prompt = f"""Analyze this email content and provide a detailed analysis.
        Focus on:
        - Email intent and urgency
        - Sentiment analysis
        - Required actions
        - Risk assessment
        - Priority level

        You MUST respond with ONLY a JSON object in this exact format:
        {{
            "intent": {{
                "primary_type": "request|complaint|inquiry|notification|other",
                "urgency": "high|medium|low",
                "confidence": <float between 0.0-1.0>
            }},
            "sentiment": {{
                "type": "positive|negative|neutral",
                "score": <float between -1.0 and 1.0>
            }},
            "key_points": ["point1", "point2"],
            "risk_assessment": {{
                "level": "high|medium|low",
                "factors": ["factor1", "factor2"]
            }},
            "suggested_actions": [
                {{
                    "type": "action type",
                    "priority": "high|medium|low",
                    "description": "action description",
                    "reason": "reason for action"
                }}
            ]
        }}

        Subject: {email_data['subject']}
        From: {email_data['from']}
        To: {email_data['to']}
        Body:
        ---
        {email_data['body'][:3000]}
        ---"""

        try:
            # Generate content with Google Gemini model (not awaitable)
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.3,
                    'top_p': 0.8,
                    'top_k': 40
                }
            )
            
            # Extract the response text
            response_text = response.text.strip()
            
            # Parse the JSON response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response if it contains additional text
                json_match = re.search(r'({.*})', response_text.replace('\n', ' '))
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    # If JSON parsing fails, return a fallback analysis
                    logger.warning(f"Failed to parse JSON from response: {response_text[:100]}...")
                    return self._generate_fallback_analysis()
                
            return self._validate_analysis_result(result)
            
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            return self._generate_fallback_analysis()
            
    def _validate_analysis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean analysis result."""
        # Validate intent
        if not isinstance(result.get("intent"), dict):
            result["intent"] = {
                "primary_type": "other",
                "urgency": "medium",
                "confidence": 0.5
            }
        
        # Validate sentiment
        if not isinstance(result.get("sentiment"), dict):
            result["sentiment"] = {
                "type": "neutral",
                "score": 0.0
            }
        
        # Validate key points
        if not isinstance(result.get("key_points"), list):
            result["key_points"] = []
        
        # Validate risk assessment
        if not isinstance(result.get("risk_assessment"), dict):
            result["risk_assessment"] = {
                "level": "medium",
                "factors": []
            }
        
        # Validate suggested actions
        if not isinstance(result.get("suggested_actions"), list):
            result["suggested_actions"] = []
        
        return result
            
    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract entities from email content."""
        entities = {
            "email_addresses": self._extract_emails(content),
            "phone_numbers": self._extract_phone_numbers(content),
            "dates": self._extract_dates(content),
            "monetary_values": self._extract_monetary_values(content),
            "urls": self._extract_urls(content)
        }
        return entities
        
    def _extract_emails(self, content: str) -> List[str]:
        """Extract email addresses using regex."""
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return list(set(re.findall(pattern, content)))
        
    def _extract_phone_numbers(self, content: str) -> List[str]:
        """Extract phone numbers using regex."""
        pattern = r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        return list(set(re.findall(pattern, content)))
        
    def _extract_dates(self, content: str) -> List[str]:
        """Extract dates using regex."""
        pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}'
        return list(set(re.findall(pattern, content, re.IGNORECASE)))
        
    def _extract_monetary_values(self, content: str) -> List[str]:
        """Extract monetary values using regex."""
        pattern = r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)'
        return list(set(re.findall(pattern, content)))
        
    def _extract_urls(self, content: str) -> List[str]:
        """Extract URLs using regex."""
        pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        return list(set(re.findall(pattern, content)))
        
    def _generate_actions(self, content_analysis: Dict[str, Any], email_data: Dict[str, Any], entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate actions based on email analysis."""
        actions = []
        
        # Add actions from content analysis
        if "suggested_actions" in content_analysis:
            actions.extend(content_analysis["suggested_actions"])
        
        # Add intent-based actions
        intent = content_analysis.get("intent", {})
        if intent.get("primary_type") == "complaint":
            actions.append({
                "type": "customer_support",
                "priority": "high",
                "description": "Escalate to customer support",
                "reason": "Email identified as customer complaint"
            })
        elif intent.get("primary_type") == "request":
            actions.append({
                "type": "request_processing",
                "priority": "high" if intent.get("urgency") == "high" else "medium",
                "description": "Process customer request",
                "reason": "Email contains customer request"
            })
        
        # Add urgency-based actions
        if intent.get("urgency") == "high":
            actions.append({
                "type": "urgent_response",
                "priority": "high",
                "description": "Send immediate response",
                "reason": "High urgency email requires quick response"
            })
        
        # Add sentiment-based actions
        sentiment = content_analysis.get("sentiment", {})
        if sentiment.get("type") == "negative" and sentiment.get("score", 0) < -0.5:
            actions.append({
                "type": "sentiment_alert",
                "priority": "high",
                "description": "Address negative sentiment",
                "reason": "Strong negative sentiment detected"
            })
        
        # Add risk-based actions
        risk = content_analysis.get("risk_assessment", {})
        if risk.get("level") == "high":
            actions.append({
                "type": "risk_assessment",
                "priority": "high",
                "description": "Evaluate and mitigate risks",
                "reason": f"High risk factors: {', '.join(risk.get('factors', []))}"
            })
        
        # Add attachment-based actions
        if email_data.get("attachments"):
            actions.append({
                "type": "attachment_review",
                "priority": "medium",
                "description": "Review email attachments",
                "reason": f"Email contains {len(email_data['attachments'])} attachment(s)"
            })
        
        # Add entity-based actions
        if entities.get("monetary_values"):
            actions.append({
                "type": "financial_review",
                "priority": "high",
                "description": "Review financial implications",
                "reason": "Email contains monetary values"
            })
        
        # Deduplicate actions
        seen = set()
        unique_actions = []
        for action in actions:
            action_key = f"{action['type']}_{action['priority']}"
            if action_key not in seen:
                seen.add(action_key)
                unique_actions.append(action)
        
        return unique_actions
        
    def _generate_fallback_analysis(self) -> Dict[str, Any]:
        """Generate fallback analysis when main analysis fails."""
        return {
            "intent": {
                "primary_type": "other",
                "urgency": "medium",
                "confidence": 0.5
            },
            "sentiment": {
                "type": "neutral",
                "score": 0.0
            },
            "key_points": ["Manual review recommended"],
            "risk_assessment": {
                "level": "medium",
                "factors": ["Automated analysis failed"]
            },
            "suggested_actions": [{
                "type": "manual_review",
                "priority": "high",
                "description": "Review email manually",
                "reason": "Automated analysis failed"
            }]
        } 