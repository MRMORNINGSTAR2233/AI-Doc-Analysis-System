import os
import google.generativeai as genai
from pathlib import Path
import json
import logging
from typing import Dict, Any, List
import mimetypes
from email import message_from_file
import pypdf
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class ClassifierAgent:
    def __init__(self):
        # Initialize Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Define business intents with comprehensive patterns
        self.business_intents = {
            "COMPLAINT": {
                "keywords": ["dissatisfied", "angry", "issue", "problem", "refund", "complaint", 
                           "broken", "failed", "poor", "terrible", "fix", "resolve"],
                "patterns": [
                    r"(?i)(?:not? ?(?:work|happy|satisfied))",
                    r"(?i)(?:problems?|issues?|concerns?)",
                    r"(?i)(?:complain|dissatisf|disappoint)",
                    r"(?i)(?:refund|return|money back)"
                ]
            },
            "INVOICE": {
                "keywords": ["invoice", "payment", "amount", "due", "bill", "cost", "price",
                           "charge", "paid", "balance", "total", "receipt"],
                "patterns": [
                    r"(?i)(?:\$\s*\d+(?:,\d{3})*(?:\.\d{2})?)",
                    r"(?i)(?:invoice|bill|payment).*?(?:number|#).*?(\d+)",
                    r"(?i)(?:total|amount|sum).*?(?:due|payable)",
                    r"(?i)(?:paid?|payment).*?(?:by|before|due)"
                ]
            },
            "RFQ": {
                "keywords": ["quote", "proposal", "pricing", "estimate", "cost", "inquiry",
                           "specifications", "requirements", "scope", "project"],
                "patterns": [
                    r"(?i)(?:request.*?(?:quote|proposal|estimate))",
                    r"(?i)(?:price.*?(?:inquiry|request|quote))",
                    r"(?i)(?:looking.*?(?:for|to).*?(?:quote|estimate))",
                    r"(?i)(?:specifications?|requirements?|scope)"
                ]
            },
            "REGULATION": {
                "keywords": ["compliance", "GDPR", "FDA", "regulation", "policy", "legal",
                           "requirement", "standard", "guideline", "protocol"],
                "patterns": [
                    r"(?i)(?:comply|compliance|regulatory)",
                    r"(?i)(?:GDPR|HIPAA|FDA|SOX|PCI)",
                    r"(?i)(?:regulation|policy|standard|requirement)",
                    r"(?i)(?:legal|law|statute|directive)"
                ]
            },
            "FRAUD_RISK": {
                "keywords": ["suspicious", "fraud", "unauthorized", "alert", "risk", "security",
                           "breach", "violation", "unusual", "investigate"],
                "patterns": [
                    r"(?i)(?:fraud|suspicious|unauthorized)",
                    r"(?i)(?:security.*?(?:breach|incident|violation))",
                    r"(?i)(?:unusual|irregular|anomaly)",
                    r"(?i)(?:investigate|verify|validate)"
                ]
            }
        }
        
    async def classify(self, file_path: Path) -> Dict[str, Any]:
        """Classify document format and intent."""
        try:
            # Detect format
            format_type = await self._detect_format(file_path)
            
            # Extract content
            content = await self._extract_content(file_path, format_type)
            if not content.strip():
                raise ValueError("Empty document content")
                
            # Analyze content using multiple methods
            analysis_results = await self._analyze_content_multiple_methods(content)
            
            # Combine and validate results
            final_result = self._combine_analysis_results(analysis_results)
            
            # Add metadata and additional analysis
            result = {
                "format": format_type,
                "intent": {
                    "type": final_result["primary_intent"],
                    "confidence": final_result["confidence"],
                    "keywords": final_result["keywords"],
                    "reasoning": final_result["reasoning"],
                    "analysis_method": final_result["method"]
                },
                "metadata": self._get_file_metadata(file_path),
                "analysis": self._generate_analysis(final_result, content),
                "routing": self._generate_routing(final_result)
            }
            
            logger.info(f"Classification complete: {result['format']} - {result['intent']['type']} ({result['intent']['confidence']:.1%})")
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return self._generate_fallback_response(file_path, str(e))
            
    async def _analyze_content_multiple_methods(self, content: str) -> List[Dict[str, Any]]:
        """Analyze content using multiple methods for robust classification."""
        results = []
        
        # Method 1: LLM-based analysis
        try:
            llm_result = await self._detect_intent_llm(content)
            if self._is_valid_analysis(llm_result):
                results.append({**llm_result, "method": "llm"})
        except Exception as e:
            logger.warning(f"LLM analysis failed: {str(e)}")
            
        # Method 2: Pattern-based analysis
        try:
            pattern_result = self._detect_intent_patterns(content)
            results.append({**pattern_result, "method": "pattern"})
        except Exception as e:
            logger.warning(f"Pattern analysis failed: {str(e)}")
            
        # Method 3: Keyword-based analysis
        try:
            keyword_result = self._detect_intent_keywords(content)
            results.append({**keyword_result, "method": "keyword"})
        except Exception as e:
            logger.warning(f"Keyword analysis failed: {str(e)}")
            
        return results
        
    async def _detect_intent_llm(self, content: str) -> Dict[str, Any]:
        """Detect intent using LLM with improved prompt."""
        prompt = f"""Analyze this business document content and determine its intent.
        Focus on key indicators and provide a confident classification.
        
        Available intent types:
        - COMPLAINT: Customer complaints, issues, or dissatisfaction
        - INVOICE: Bills, payments, financial transactions
        - RFQ: Requests for quotes, proposals, or pricing
        - REGULATION: Compliance, legal, or regulatory matters
        - FRAUD_RISK: Security concerns, fraud alerts, or risks
        
        Respond with ONLY a JSON object in this format:
        {{
            "primary_intent": "COMPLAINT|INVOICE|RFQ|REGULATION|FRAUD_RISK",
            "confidence": <float 0.1-1.0>,
            "keywords": ["key1", "key2"],
            "reasoning": "Clear explanation of classification"
        }}
        
        Content to analyze:
        ---
        {content[:2000]}
        ---"""
        
        try:
            response = await self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.3,
                    'top_p': 0.8,
                    'top_k': 40
                }
            )
            
            # Parse and validate response
            result = json.loads(response.text.strip())
            return self._validate_intent_result(result)
            
        except Exception as e:
            logger.error(f"LLM intent detection failed: {str(e)}")
            raise
            
    def _detect_intent_patterns(self, content: str) -> Dict[str, Any]:
        """Detect intent using regex patterns."""
        scores = {intent: 0.0 for intent in self.business_intents}
        matches = {intent: [] for intent in self.business_intents}
        
        # Check each intent's patterns
        for intent, config in self.business_intents.items():
            for pattern in config["patterns"]:
                found = re.findall(pattern, content)
                if found:
                    scores[intent] += len(found) * 0.2  # Weight for pattern matches
                    matches[intent].extend(found)
                    
        # Find best match
        max_score = max(scores.values())
        if max_score == 0:
            return self._generate_default_intent()
            
        primary_intent = max(scores.items(), key=lambda x: x[1])[0]
        confidence = min(0.9, 0.3 + (max_score * 0.1))
        
        return {
            "primary_intent": primary_intent,
            "confidence": confidence,
            "keywords": list(set(matches[primary_intent])),
            "reasoning": f"Found {len(matches[primary_intent])} pattern matches for {primary_intent}"
        }
        
    def _detect_intent_keywords(self, content: str) -> Dict[str, Any]:
        """Detect intent using keyword matching."""
        content_lower = content.lower()
        scores = {intent: 0.0 for intent in self.business_intents}
        matches = {intent: [] for intent in self.business_intents}
        
        # Check each intent's keywords
        for intent, config in self.business_intents.items():
            for keyword in config["keywords"]:
                if keyword.lower() in content_lower:
                    scores[intent] += 0.15  # Weight for keyword matches
                    matches[intent].append(keyword)
                    
        # Find best match
        max_score = max(scores.values())
        if max_score == 0:
            return self._generate_default_intent()
            
        primary_intent = max(scores.items(), key=lambda x: x[1])[0]
        confidence = min(0.85, 0.3 + (max_score * 0.1))
        
        return {
            "primary_intent": primary_intent,
            "confidence": confidence,
            "keywords": list(set(matches[primary_intent])),
            "reasoning": f"Found {len(matches[primary_intent])} keyword matches for {primary_intent}"
        }
        
    def _combine_analysis_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple analysis methods."""
        if not results:
            return self._generate_default_intent()
            
        # Weight the results by method reliability
        method_weights = {
            "llm": 0.5,
            "pattern": 0.3,
            "keyword": 0.2
        }
        
        # Calculate weighted scores for each intent
        intent_scores = {intent: 0.0 for intent in self.business_intents}
        
        for result in results:
            weight = method_weights.get(result["method"], 0.1)
            intent_scores[result["primary_intent"]] += result["confidence"] * weight
            
        # Select primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        # Combine keywords and reasoning
        all_keywords = []
        all_reasoning = []
        
        for result in results:
            if result["primary_intent"] == primary_intent:
                all_keywords.extend(result.get("keywords", []))
                all_reasoning.append(f"{result['method'].upper()}: {result['reasoning']}")
                
        # Calculate final confidence
        confidence = min(0.95, max(0.1, intent_scores[primary_intent]))
        
        return {
            "primary_intent": primary_intent,
            "confidence": confidence,
            "keywords": list(set(all_keywords)),
            "reasoning": " | ".join(all_reasoning),
            "method": "combined",
            "method_scores": {r["method"]: r["confidence"] for r in results}
        }
        
    def _generate_default_intent(self) -> Dict[str, Any]:
        """Generate a default intent when analysis fails."""
        return {
            "primary_intent": "RFQ",
            "confidence": 0.3,
            "keywords": [],
            "reasoning": "Default classification due to insufficient indicators",
            "method": "default"
        }
        
    def _validate_intent_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean up intent analysis result."""
        valid_intents = set(self.business_intents.keys())
        
        # Ensure valid primary intent
        if not isinstance(result.get("primary_intent"), str) or result["primary_intent"] not in valid_intents:
            result["primary_intent"] = "RFQ"
            
        # Ensure valid confidence
        try:
            result["confidence"] = float(result.get("confidence", 0.3))
            result["confidence"] = max(0.1, min(1.0, result["confidence"]))
        except (TypeError, ValueError):
            result["confidence"] = 0.3
            
        # Ensure valid keywords
        if not isinstance(result.get("keywords"), list):
            result["keywords"] = []
        result["keywords"] = [str(k) for k in result["keywords"] if k]
        
        # Ensure valid reasoning
        if not isinstance(result.get("reasoning"), str) or not result["reasoning"]:
            result["reasoning"] = f"Document classified as {result['primary_intent']}"
            
        return result
        
    async def _extract_content(self, file_path: Path, format_type: str) -> str:
        """Extract text content from file with format-specific handling."""
        try:
            if format_type == "pdf":
                return await self._extract_pdf_content(file_path)
            elif format_type == "email":
                return await self._extract_email_content(file_path)
            elif format_type == "json":
                return await self._extract_json_content(file_path)
            else:
                return await self._extract_text_content(file_path)
        except Exception as e:
            logger.error(f"Content extraction failed: {str(e)}")
            raise
            
    async def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract content from PDF with enhanced error handling."""
        try:
            reader = pypdf.PdfReader(file_path)
            text_parts = []
            
            for page in reader.pages:
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from PDF page: {str(e)}")
                    
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise
            
    async def _extract_email_content(self, file_path: Path) -> str:
        """Extract content from email with enhanced parsing."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                msg = message_from_file(f)
                
            parts = []
            
            # Add subject
            if msg.get('subject'):
                parts.append(f"Subject: {msg['subject']}")
                
            # Add from/to
            if msg.get('from'):
                parts.append(f"From: {msg['from']}")
            if msg.get('to'):
                parts.append(f"To: {msg['to']}")
                
            # Add body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            text = part.get_payload(decode=True).decode()
                            parts.append(text)
                        except:
                            continue
            else:
                try:
                    text = msg.get_payload(decode=True).decode()
                    parts.append(text)
                except:
                    text = msg.get_payload()
                    parts.append(text)
                    
            return "\n\n".join(parts)
        except Exception as e:
            logger.error(f"Email extraction failed: {str(e)}")
            raise
            
    async def _extract_json_content(self, file_path: Path) -> str:
        """Extract content from JSON with formatting."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"JSON extraction failed: {str(e)}")
            raise
            
    async def _extract_text_content(self, file_path: Path) -> str:
        """Extract content from text file with encoding handling."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Text extraction failed with {encoding}: {str(e)}")
                continue
                
        raise ValueError("Failed to extract text content with any encoding")
        
    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get enhanced file metadata."""
        try:
            return {
                "filename": file_path.name,
                "size": os.path.getsize(file_path),
                "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                "extension": file_path.suffix.lower(),
                "path": str(file_path)
            }
        except Exception as e:
            logger.warning(f"Error getting file metadata: {str(e)}")
            return {
                "filename": file_path.name,
                "error": str(e)
            }
            
    def _generate_analysis(self, intent_result: Dict[str, Any], content: str) -> Dict[str, Any]:
        """Generate comprehensive analysis results."""
        return {
            "priority_level": self._calculate_priority(intent_result),
            "requires_immediate_attention": self._needs_immediate_attention(intent_result),
            "suggested_actions": self._get_suggested_actions(intent_result),
            "key_entities": self._extract_key_entities(content),
            "summary": self._generate_summary(content, intent_result)
        }
        
    def _generate_routing(self, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate routing information."""
        return {
            "suggested_department": self._get_suggested_department(intent_result),
            "suggested_priority": "high" if intent_result["confidence"] > 0.8 else "medium",
            "requires_human_review": (
                intent_result["primary_intent"] in ["COMPLAINT", "FRAUD_RISK", "REGULATION"]
                or intent_result["confidence"] < 0.6
            )
        }
        
    def _calculate_priority(self, intent_result: Dict[str, Any]) -> str:
        """Calculate priority level based on intent and confidence."""
        if intent_result["primary_intent"] in ["FRAUD_RISK", "COMPLAINT"]:
            return "high"
        elif intent_result["confidence"] > 0.8:
            return "high"
        elif intent_result["confidence"] > 0.5:
            return "medium"
        else:
            return "low"
            
    def _needs_immediate_attention(self, intent_result: Dict[str, Any]) -> bool:
        """Determine if immediate attention is required."""
        return (
            intent_result["primary_intent"] in ["FRAUD_RISK", "COMPLAINT"]
            and intent_result["confidence"] > 0.7
        )

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
            
    def _is_valid_analysis(self, result: Dict[str, Any]) -> bool:
        """Validate analysis result."""
        if not isinstance(result, dict):
            return False
            
        required_fields = {
            "primary_intent": str,
            "confidence": (int, float),
            "keywords": list,
            "reasoning": str
        }
        
        # Check required fields and types
        for field, field_type in required_fields.items():
            if field not in result:
                return False
            if not isinstance(result[field], field_type):
                return False
                
        # Validate intent
        if result["primary_intent"] not in self.business_intents:
            return False
            
        # Validate confidence
        if not (0 < float(result["confidence"]) <= 1):
            return False
            
        return True
        
    def _generate_fallback_response(self, file_path: Path, error_msg: str) -> Dict[str, Any]:
        """Generate a complete fallback response."""
        return {
            "format": "unknown",
            "intent": {
                "type": "RFQ",
                "confidence": 0.3,
                "keywords": [],
                "reasoning": f"Classification failed: {error_msg}",
                "analysis_method": "fallback"
            },
            "metadata": self._get_file_metadata(file_path),
            "analysis": {
                "priority_level": "low",
                "requires_immediate_attention": False,
                "suggested_actions": [
                    {
                        "type": "manual_review",
                        "priority": "high",
                        "reason": "Automated classification failed"
                    }
                ],
                "key_entities": {},
                "summary": "Document analysis failed, manual review required"
            },
            "routing": {
                "suggested_department": "General Processing",
                "suggested_priority": "medium",
                "requires_human_review": True
            }
        }

    def _get_suggested_actions(self, intent_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggested actions based on intent analysis."""
        actions = []
        
        if intent_analysis["primary_intent"] == "COMPLAINT":
            actions.extend([
                {"type": "escalate", "priority": "high", "reason": "Customer complaint requires immediate attention"},
                {"type": "notify_support", "priority": "high", "reason": "Alert customer support team"},
                {"type": "create_case", "priority": "high", "reason": "Create support case for tracking"}
            ])
        elif intent_analysis["primary_intent"] == "FRAUD_RISK":
            actions.extend([
                {"type": "risk_assessment", "priority": "high", "reason": "Potential fraud detected"},
                {"type": "notify_security", "priority": "high", "reason": "Alert security team"},
                {"type": "hold_transaction", "priority": "high", "reason": "Prevent further processing"}
            ])
        elif intent_analysis["primary_intent"] == "INVOICE":
            actions.extend([
                {"type": "verify_amount", "priority": "medium", "reason": "Validate invoice details"},
                {"type": "route_approval", "priority": "medium", "reason": "Route for payment approval"}
            ])
        
        return actions

    def _get_suggested_department(self, intent_analysis: Dict[str, Any]) -> str:
        """Determine suggested department based on intent."""
        intent_to_dept = {
            "COMPLAINT": "Customer Support",
            "INVOICE": "Finance",
            "RFQ": "Sales",
            "REGULATION": "Legal",
            "FRAUD_RISK": "Risk Management"
        }
        return intent_to_dept.get(intent_analysis["primary_intent"], "General Processing")

    def _extract_key_entities(self, content: str) -> Dict[str, Any]:
        """Extract key entities from content."""
        entities = {
            "monetary_values": self._extract_monetary_values(content),
            "dates": self._extract_dates(content),
            "email_addresses": self._extract_emails(content),
            "phone_numbers": self._extract_phone_numbers(content)
        }
        return entities

    def _extract_monetary_values(self, content: str) -> List[str]:
        """Extract monetary values using regex."""
        pattern = r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)'
        return list(set(re.findall(pattern, content)))

    def _extract_dates(self, content: str) -> List[str]:
        """Extract dates using regex."""
        pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}'
        return list(set(re.findall(pattern, content, re.IGNORECASE)))

    def _extract_emails(self, content: str) -> List[str]:
        """Extract email addresses using regex."""
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return list(set(re.findall(pattern, content)))

    def _extract_phone_numbers(self, content: str) -> List[str]:
        """Extract phone numbers using regex."""
        pattern = r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        return list(set(re.findall(pattern, content)))

    def _generate_summary(self, content: str, intent_analysis: Dict[str, Any]) -> str:
        """Generate a brief summary based on content and intent."""
        summary = f"This appears to be a {intent_analysis['primary_intent'].lower()} document "
        summary += f"with {len(intent_analysis['keywords'])} key indicators. "
        summary += intent_analysis['reasoning']
        return summary 