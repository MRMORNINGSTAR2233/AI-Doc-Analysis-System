import os
import google.generativeai as genai
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Set
import mimetypes
from email import message_from_file
import pypdf
import re
from datetime import datetime
import numpy as np
from collections import defaultdict
import hashlib
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass
from enum import Enum
import stat

logger = logging.getLogger(__name__)

class AnalysisMethod(Enum):
    LLM = "llm"
    PATTERN = "pattern"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    FALLBACK = "fallback"

@dataclass
class AnalysisResult:
    primary_intent: str
    confidence: float
    keywords: List[str]
    reasoning: str
    method: AnalysisMethod
    metadata: Dict[str, Any] = None

class DocumentType(Enum):
    PDF = "pdf"
    EMAIL = "email"
    JSON = "json"
    TEXT = "text"
    UNKNOWN = "unknown"

class ClassifierAgent:
    def __init__(self):
        # Initialize Google Gemini with enhanced configuration
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Enhanced business intents with weighted patterns and advanced regex
        self.business_intents = self._initialize_business_intents()
        
        # Initialize content cache
        self.content_cache = {}
        
        # Configure advanced settings
        self.config = {
            'min_confidence_threshold': 0.4,
            'cache_ttl': 3600,  # 1 hour
            'max_retries': 3,
            'concurrent_analysis': True,
            'enable_advanced_features': True
        }

    def _initialize_business_intents(self) -> Dict[str, Dict[str, Any]]:
        """Initialize enhanced business intents with weighted patterns and metadata."""
        base_intents = {
            "COMPLAINT": {
                "keywords": [
                    ("dissatisfied", 1.0), ("angry", 0.9), ("issue", 0.8),
                    ("problem", 0.8), ("refund", 1.0), ("complaint", 1.0),
                    ("broken", 0.9), ("failed", 0.8), ("poor", 0.7),
                    ("terrible", 0.9), ("fix", 0.7), ("resolve", 0.7),
                    ("unacceptable", 1.0), ("disappointed", 0.9)
                ],
                "patterns": [
                    (r"(?i)(?:not?\s+(?:work|happy|satisfied))", 1.0),
                    (r"(?i)(?:problems?|issues?|concerns?)\s+with", 0.9),
                    (r"(?i)(?:complain|dissatisf|disappoint).*?(?:service|product|quality)", 1.0),
                    (r"(?i)(?:refund|return|money\s+back).*?(?:demand|request|want)", 1.0),
                    (r"(?i)(?:escalate|urgent|immediate).*?(?:attention|response|action)", 0.9)
                ],
                "metadata": {
                    "priority": "high",
                    "response_time": "4h",
                    "requires_human": True
                }
            },
            "INVOICE": {
                "keywords": [
                    ("invoice", 1.0), ("payment", 0.9), ("amount", 0.8),
                    ("due", 0.8), ("bill", 0.9), ("cost", 0.7),
                    ("price", 0.7), ("charge", 0.8), ("paid", 0.8),
                    ("balance", 0.7), ("total", 0.8), ("receipt", 0.9),
                    ("tax", 0.8), ("subtotal", 0.9)
                ],
                "patterns": [
                    (r"(?i)(?:invoice|bill)\s*#?\s*\d+", 1.0),
                    (r"(?i)(?:total|amount|sum).*?(?:due|payable):?\s*[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?", 1.0),
                    (r"(?i)(?:payment\s+terms?|due\s+date):?\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", 0.9),
                    (r"(?i)(?:tax|vat)\s+(?:id|number):?\s*[A-Z0-9-]+", 0.8)
                ],
                "metadata": {
                    "priority": "medium",
                    "response_time": "24h",
                    "requires_human": False
                }
            },
            "RFQ": {
                "keywords": [
                    ("quote", 1.0), ("proposal", 0.9), ("pricing", 0.8),
                    ("estimate", 0.9), ("inquiry", 0.8), ("specifications", 0.7),
                    ("requirements", 0.8), ("scope", 0.7), ("project", 0.6)
                ],
                "patterns": [
                    (r"(?i)(?:request.*?(?:quote|proposal|estimate))", 1.0),
                    (r"(?i)(?:price.*?(?:inquiry|request|quote))", 0.9),
                    (r"(?i)(?:looking.*?(?:for|to).*?(?:quote|estimate))", 0.8),
                    (r"(?i)(?:specifications?|requirements?|scope)", 0.7)
                ],
                "metadata": {
                    "priority": "medium",
                    "response_time": "48h",
                    "requires_human": True
                }
            },
            "REGULATION": {
                "keywords": [
                    ("compliance", 1.0), ("regulation", 1.0), ("policy", 0.8),
                    ("legal", 0.9), ("requirement", 0.8), ("standard", 0.7),
                    ("guideline", 0.7), ("protocol", 0.8), ("directive", 0.9)
                ],
                "patterns": [
                    (r"(?i)(?:comply|compliance|regulatory)", 1.0),
                    (r"(?i)(?:GDPR|HIPAA|FDA|SOX|PCI)", 1.0),
                    (r"(?i)(?:regulation|policy|standard|requirement)", 0.9),
                    (r"(?i)(?:legal|law|statute|directive)", 0.8)
                ],
                "metadata": {
                    "priority": "high",
                    "response_time": "24h",
                    "requires_human": True
                }
            },
            "FRAUD_RISK": {
                "keywords": [
                    ("fraud", 1.0), ("suspicious", 0.9), ("unauthorized", 1.0),
                    ("security", 0.8), ("breach", 1.0), ("violation", 0.9),
                    ("alert", 0.8), ("investigate", 0.7), ("risk", 0.8)
                ],
                "patterns": [
                    (r"(?i)(?:fraud|suspicious|unauthorized)", 1.0),
                    (r"(?i)(?:security.*?(?:breach|incident|violation))", 1.0),
                    (r"(?i)(?:unusual|irregular|anomaly)", 0.9),
                    (r"(?i)(?:investigate|verify|validate)", 0.8)
                ],
                "metadata": {
                    "priority": "critical",
                    "response_time": "1h",
                    "requires_human": True
                }
            }
        }
        
        return base_intents

    async def classify(self, file_path: Path) -> Dict[str, Any]:
        """Enhanced document classification with advanced error handling and validation."""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(file_path)
            if cached_result := self.content_cache.get(cache_key):
                if self._is_cache_valid(cached_result):
                    return cached_result['result']

            # Detect format with enhanced validation
            format_type = await self._detect_format_advanced(file_path)
            
            # Handle unknown format
            if format_type == DocumentType.UNKNOWN:
                raise ValueError(f"Unsupported file format for: {file_path.name}")
            
            # Extract content with advanced processing
            content = await self._extract_content_advanced(file_path, format_type.value)
            if not content.strip():
                raise ValueError("Empty document content")
                
            # Perform concurrent analysis if enabled
            if self.config['concurrent_analysis']:
                analysis_results = await self._analyze_content_concurrent(content)
            else:
                analysis_results = await self._analyze_content_sequential(content)
            
            # Enhanced result combination with weighted scoring
            final_result = self._combine_analysis_results_advanced(analysis_results)
            
            # Add comprehensive metadata and analysis
            result = {
                "format": format_type.value,
                "intent": self._enhance_intent_result(final_result),
                "metadata": await self._get_enhanced_metadata(file_path),
                "analysis": await self._generate_advanced_analysis(final_result, content),
                "routing": self._generate_enhanced_routing(final_result),
                "validation": self._validate_result_comprehensive(final_result)
            }
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            logger.info(f"Advanced classification complete: {result['format']} - {result['intent']['type']} ({result['intent']['confidence']:.1%})")
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}", exc_info=True)
            return await self._generate_advanced_fallback(file_path, str(e))

    async def _analyze_content_sequential(self, content: str) -> List[AnalysisResult]:
        """Perform sequential content analysis using multiple methods."""
        try:
            # Start with pattern and keyword analysis (faster methods)
            pattern_result = await self._detect_intent_patterns_advanced(content)
            keyword_result = await self._detect_intent_keywords_advanced(content)
            
            # If both have high confidence, skip LLM to save time
            if pattern_result.confidence > 0.7 and keyword_result.confidence > 0.7:
                return [pattern_result, keyword_result]
            
            # Otherwise, add LLM and hybrid analysis
            llm_result = await self._detect_intent_llm(content)
            hybrid_result = await self._detect_intent_hybrid(content)
            
            return [pattern_result, keyword_result, llm_result, hybrid_result]
        except Exception as e:
            logger.error(f"Sequential analysis failed: {str(e)}")
            return [self._generate_fallback_analysis()]

    async def _analyze_content_concurrent(self, content: str) -> List[AnalysisResult]:
        """Perform concurrent content analysis using multiple methods."""
        try:
            results = []
            
            # Pattern analysis
            pattern_result = await self._detect_intent_patterns_advanced(content)
            results.append(pattern_result)
            
            # Keyword analysis
            keyword_result = await self._detect_intent_keywords_advanced(content)
            results.append(keyword_result)
            
            # LLM analysis (optional based on previous results)
            if pattern_result.confidence < 0.7 or keyword_result.confidence < 0.7:
                llm_result = await self._detect_intent_llm(content)
                results.append(llm_result)
            
            # Hybrid analysis
            hybrid_result = await self._detect_intent_hybrid(content)
            results.append(hybrid_result)
            
            return results
        except Exception as e:
            logger.error(f"Concurrent analysis failed: {str(e)}")
            return [self._generate_fallback_analysis()]

    async def _detect_intent_llm(self, content: str) -> AnalysisResult:
        """Detect intent using LLM with improved prompt."""
        try:
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
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.3,
                        'top_p': 0.8,
                        'top_k': 40
                    }
                )
                
                # Extract text directly from response
                response_text = response.text.strip()
                
                # Handle potential JSON-parsing errors
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from response if it contains additional text
                    json_match = re.search(r'({.*})', response_text.replace('\n', ' '))
                    if json_match:
                        result = json.loads(json_match.group(1))
                    else:
                        # Fallback to manual intent detection
                        if "complaint" in response_text.lower():
                            return AnalysisResult(
                                primary_intent="COMPLAINT",
                                confidence=0.6,
                                keywords=["complaint", "issue"],
                                reasoning="Extracted from non-JSON response",
                                method=AnalysisMethod.LLM
                            )
                        elif "invoice" in response_text.lower():
                            return AnalysisResult(
                                primary_intent="INVOICE",
                                confidence=0.6,
                                keywords=["invoice", "payment"],
                                reasoning="Extracted from non-JSON response",
                                method=AnalysisMethod.LLM
                            )
                        else:
                            return AnalysisResult(
                                primary_intent="RFQ",
                                confidence=0.5,
                                keywords=["request"],
                                reasoning="Default fallback from non-JSON response",
                                method=AnalysisMethod.LLM
                            )
                
                # Validate the result
                if not isinstance(result, dict):
                    raise ValueError("Invalid response format")
                    
                required_fields = {
                    "primary_intent": str,
                    "confidence": (int, float),
                    "keywords": list,
                    "reasoning": str
                }
                
                for field, field_type in required_fields.items():
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")
                    if not isinstance(result[field], field_type):
                        raise ValueError(f"Invalid type for field {field}")
                        
                # Ensure valid intent
                if result["primary_intent"] not in self.business_intents:
                    raise ValueError(f"Invalid intent: {result['primary_intent']}")
                    
                # Ensure valid confidence
                result["confidence"] = float(result["confidence"])
                if not (0 < result["confidence"] <= 1):
                    raise ValueError(f"Invalid confidence value: {result['confidence']}")
                    
                return AnalysisResult(
                    primary_intent=result["primary_intent"],
                    confidence=result["confidence"],
                    keywords=result["keywords"],
                    reasoning=result["reasoning"],
                    method=AnalysisMethod.LLM
                )
            except Exception as inner_e:
                # If model call fails, fall back to pattern analysis
                logger.error(f"LLM generation failed: {str(inner_e)}")
                pattern_result = await self._detect_intent_patterns_advanced(content)
                pattern_result.reasoning = f"LLM analysis failed, using pattern fallback: {pattern_result.reasoning}"
                return pattern_result
                
        except Exception as e:
            logger.error(f"LLM intent detection failed: {str(e)}")
            return self._generate_fallback_analysis()

    def _get_confidence_level(self, confidence: float) -> str:
        """Convert numerical confidence to descriptive level."""
        if confidence >= 0.9:
            return "very high"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.6:
            return "moderate"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very low"
            
    async def _generate_advanced_analysis(self, result: AnalysisResult, content: str) -> Dict[str, Any]:
        """Generate comprehensive analysis results with advanced metrics."""
        try:
            return {
                "classification": {
                    "primary_intent": result.primary_intent,
                    "confidence": result.confidence,
                    "confidence_level": self._get_confidence_level(result.confidence),
                    "method": result.method.value,
                    "competing_intents": self._find_competing_intents(result)
                },
                "content_analysis": {
                    "length": len(content),
                    "structure": self._analyze_text_structure(content),
                    "readability": self._calculate_readability(content),
                    "sentiment": await self._analyze_sentiment(content),
                    "key_phrases": self._extract_key_phrases(content),
                    "entities": self._extract_entities(content)
                },
                "metadata": {
                    "processing_time": datetime.now().isoformat(),
                    "version": "2.0",
                    "model": "gemini-1.5-flash"
                },
                "recommendations": self._generate_recommendations(result)
            }
        except Exception as e:
            logger.error(f"Advanced analysis generation failed: {str(e)}")
            return {
                "classification": {
                    "primary_intent": result.primary_intent,
                    "confidence": result.confidence,
                    "confidence_level": self._get_confidence_level(result.confidence),
                    "method": result.method.value
                },
                "error": str(e),
                "fallback": True
            }
    
    def _find_competing_intents(self, result: AnalysisResult) -> List[str]:
        """Find competing intents from analysis metadata."""
        try:
            if not result.metadata or "method_scores" not in result.metadata:
                return []
                
            method_scores = result.metadata.get("method_scores", {})
            primary_intent = result.primary_intent
            competing = []
            
            for intent_name, scores in method_scores.items():
                # Skip the primary intent
                if intent_name == primary_intent:
                    continue
                    
                # Check if any score value is high enough to be competing
                if isinstance(scores, dict):
                    if any(float(score) > 0.4 for score in scores.values()):
                        competing.append(intent_name)
                elif isinstance(scores, (float, int)) and float(scores) > 0.4:
                    competing.append(intent_name)
                    
            return competing
        except Exception as e:
            logger.warning(f"Error finding competing intents: {str(e)}")
            return []
            
    def _get_suggested_department(self, result: AnalysisResult) -> str:
        """Get suggested department based on intent type."""
        try:
            intent_to_department = {
                "COMPLAINT": "Customer Support",
                "INVOICE": "Finance",
                "RFQ": "Sales",
                "REGULATION": "Legal",
                "FRAUD_RISK": "Security"
            }
            
            return intent_to_department.get(result.primary_intent, "General Processing")
        except Exception as e:
            logger.warning(f"Error getting suggested department: {str(e)}")
            return "General Processing"
            
    def _generate_recommendations(self, result: AnalysisResult) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        try:
            recommendations = []
            
            # Intent-based recommendations
            if result.primary_intent == "COMPLAINT":
                recommendations.append({
                    "type": "response",
                    "priority": "high",
                    "action": "Immediate customer response required",
                    "reason": "Customer complaint detected"
                })
            elif result.primary_intent == "FRAUD_RISK":
                recommendations.append({
                    "type": "security",
                    "priority": "critical",
                    "action": "Initiate fraud investigation protocol",
                    "reason": "Potential fraud risk detected"
                })
            elif result.primary_intent == "INVOICE":
                recommendations.append({
                    "type": "process",
                    "priority": "medium",
                    "action": "Process payment according to terms",
                    "reason": "Invoice document detected"
                })
                
            # Confidence-based recommendations
            if result.confidence < 0.6:
                recommendations.append({
                    "type": "review",
                    "priority": "high",
                    "action": "Manual review recommended due to low confidence",
                    "reason": f"Classification confidence ({result.confidence:.1%}) below threshold"
                })
                
            return recommendations
        except Exception as e:
            logger.warning(f"Error generating recommendations: {str(e)}")
            return [{
                "type": "error",
                "priority": "high",
                "action": "Review document manually",
                "reason": f"Error generating recommendations: {str(e)}"
            }]

    def _combine_analysis_results_advanced(self, results: List[AnalysisResult]) -> AnalysisResult:
        """Combine multiple analysis results with advanced weighting and validation."""
        if not results:
            return self._generate_fallback_analysis()

        # Define method weights
        method_weights = {
            AnalysisMethod.LLM: 0.5,
            AnalysisMethod.PATTERN: 0.3,
            AnalysisMethod.KEYWORD: 0.2,
            AnalysisMethod.HYBRID: 0.4,
            AnalysisMethod.FALLBACK: 0.1
        }

        # Calculate weighted scores for each intent
        intent_scores = defaultdict(float)
        intent_keywords = defaultdict(set)
        intent_reasoning = defaultdict(list)
        method_scores = defaultdict(dict)

        for result in results:
            weight = method_weights.get(result.method, 0.1)
            weighted_score = result.confidence * weight
            
            intent_scores[result.primary_intent] += weighted_score
            intent_keywords[result.primary_intent].update(result.keywords)
            intent_reasoning[result.primary_intent].append(
                f"{result.method.value}: {result.reasoning}"
            )
            method_scores[result.primary_intent][result.method.value] = result.confidence

        # Get primary intent
        if not intent_scores:
            return self._generate_fallback_analysis()

        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate final confidence
        total_weight = sum(
            method_weights[r.method] 
            for r in results 
            if r.primary_intent == primary_intent
        )
        confidence = min(0.95, intent_scores[primary_intent] / total_weight if total_weight > 0 else 0.3)

        # Combine reasoning
        combined_reasoning = " | ".join(intent_reasoning[primary_intent])
        if not combined_reasoning:
            combined_reasoning = f"Combined analysis determined {primary_intent} intent"

        # Create result with method scores
        result = AnalysisResult(
            primary_intent=primary_intent,
            confidence=confidence,
            keywords=list(intent_keywords[primary_intent]),
            reasoning=combined_reasoning,
            method=AnalysisMethod.HYBRID,
            metadata={"method_scores": method_scores[primary_intent]}
        )

        return result

    async def _detect_intent_hybrid(self, content: str) -> AnalysisResult:
        """Advanced hybrid analysis combining multiple methods."""
        try:
            # Perform initial pattern and keyword analysis
            pattern_scores = defaultdict(float)
            keyword_scores = defaultdict(float)
            
            # Get pattern scores
            for intent, config in self.business_intents.items():
                for pattern, weight in config['patterns']:
                    matches = list(re.finditer(pattern, content))
                    pattern_scores[intent] += weight * len(matches)

            # Get keyword scores
            content_lower = content.lower()
            for intent, config in self.business_intents.items():
                for keyword, weight in config['keywords']:
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    matches = list(re.finditer(pattern, content_lower))
                    keyword_scores[intent] += weight * len(matches)
            
            # Normalize scores
            total_pattern = sum(pattern_scores.values())
            total_keyword = sum(keyword_scores.values())
            
            if total_pattern > 0:
                pattern_scores = {k: v/total_pattern for k, v in pattern_scores.items()}
            if total_keyword > 0:
                keyword_scores = {k: v/total_keyword for k, v in keyword_scores.items()}
            
            # Combine scores with weighted average
            combined_scores = {}
            pattern_weight = 0.4
            keyword_weight = 0.3
            context_weight = 0.3
            
            # Use a static context score to avoid API calls
            static_context_scores = {
                "COMPLAINT": 0.7, 
                "INVOICE": 0.7,
                "RFQ": 0.7,
                "REGULATION": 0.7,
                "FRAUD_RISK": 0.7
            }
            
            for intent in self.business_intents:
                combined_scores[intent] = (
                    pattern_scores.get(intent, 0) * pattern_weight +
                    keyword_scores.get(intent, 0) * keyword_weight +
                    static_context_scores.get(intent, 0.5) * context_weight
                )
            
            # Get primary intent and confidence
            primary_intent = max(combined_scores.items(), key=lambda x: x[1])[0] if combined_scores else "RFQ"
            confidence = combined_scores.get(primary_intent, 0.3)
            
            # Extract relevant keywords
            keywords = []
            for keyword, _ in self.business_intents.get(primary_intent, {}).get('keywords', []):
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', content_lower):
                    keywords.append(keyword)
            
            # Generate reasoning
            reasoning = f"Hybrid analysis: Pattern score: {pattern_scores.get(primary_intent, 0):.2f}, " \
                        f"Keyword score: {keyword_scores.get(primary_intent, 0):.2f}"
            
            return AnalysisResult(
                primary_intent=primary_intent,
                confidence=confidence,
                keywords=keywords,
                reasoning=reasoning,
                method=AnalysisMethod.HYBRID
            )
            
        except Exception as e:
            logger.error(f"Hybrid analysis failed: {str(e)}")
            return self._generate_fallback_analysis()

    async def _calculate_context_score(self, content: str, intent: str) -> float:
        """Calculate context-based score using simplified static approach."""
        try:
            # Static scoring to avoid API calls
            intent_context_scores = {
                "COMPLAINT": 0.75, 
                "INVOICE": 0.65,
                "RFQ": 0.7,
                "REGULATION": 0.8,
                "FRAUD_RISK": 0.85
            }
            
            return intent_context_scores.get(intent, 0.5)
            
        except Exception as e:
            logger.warning(f"Context score calculation failed: {str(e)}")
            return 0.5

    async def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze text sentiment using a deterministic approach."""
        try:
            # Count sentiment-related words
            positive_words = ["good", "great", "excellent", "satisfied", "happy", "appreciate"]
            negative_words = ["bad", "poor", "issue", "problem", "dissatisfied", "complaint"]
            objective_words = ["report", "statement", "document", "invoice", "request"]
            urgent_words = ["urgent", "immediate", "critical", "asap", "emergency"]
            
            content_lower = content.lower()
            
            # Count occurrences
            positive_count = sum(content_lower.count(word) for word in positive_words)
            negative_count = sum(content_lower.count(word) for word in negative_words)
            objective_count = sum(content_lower.count(word) for word in objective_words)
            urgent_count = sum(content_lower.count(word) for word in urgent_words)
            
            # Calculate normalized scores
            total_words = len(content_lower.split())
            normalization_factor = min(1.0, total_words / 100)
            
            scores = {
                "positivity": min(1.0, positive_count * 0.1 * normalization_factor),
                "negativity": min(1.0, negative_count * 0.1 * normalization_factor),
                "objectivity": min(1.0, objective_count * 0.1 * normalization_factor),
                "urgency": min(1.0, urgent_count * 0.2 * normalization_factor)
            }
            
            return {
                "scores": scores,
                "primary_sentiment": self._determine_primary_sentiment(scores),
                "confidence": max(scores.values())
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {str(e)}")
            return {"error": str(e)}

    def _determine_primary_sentiment(self, scores: Dict[str, float]) -> str:
        """Determine primary sentiment from scores."""
        if scores["objectivity"] > 0.7:
            return "neutral"
        elif scores["positivity"] > scores["negativity"]:
            return "positive" if scores["positivity"] > 0.6 else "slightly_positive"
        else:
            return "negative" if scores["negativity"] > 0.6 else "slightly_negative"

    async def _detect_format_advanced(self, file_path: Path) -> DocumentType:
        """Enhanced file format detection with comprehensive checks."""
        try:
            # First check by extension
            extension = file_path.suffix.lower()
            extension_map = {
                '.pdf': DocumentType.PDF,
                '.json': DocumentType.JSON,
                '.eml': DocumentType.EMAIL,
                '.msg': DocumentType.EMAIL,
                '.txt': DocumentType.TEXT,
                '.doc': DocumentType.TEXT,
                '.docx': DocumentType.TEXT,
                '.rtf': DocumentType.TEXT,
                '.csv': DocumentType.TEXT,
                '.xml': DocumentType.TEXT,
                '.html': DocumentType.TEXT,
                '.htm': DocumentType.TEXT,
                '.md': DocumentType.TEXT
            }
            
            if extension in extension_map:
                # Verify the format for certain types
                if extension == '.pdf':
                    try:
                        pypdf.PdfReader(file_path)
                        return DocumentType.PDF
                    except:
                        pass
                elif extension == '.json':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json.load(f)
                        return DocumentType.JSON
                    except:
                        pass
                else:
                    return extension_map[extension]

            # Content-based detection
            try:
                # Try to read first few KB of the file
                sample_size = 8192  # 8KB sample
                with open(file_path, 'rb') as f:
                    content_sample = f.read(sample_size)

                # Check for PDF signature
                if content_sample.startswith(b'%PDF'):
                    return DocumentType.PDF

                # Check for email headers
                try:
                    content_start = content_sample.decode('utf-8', errors='ignore')
                    email_headers = ['From:', 'To:', 'Subject:', 'Date:', 'Received:', 'Message-ID:']
                    if any(header in content_start for header in email_headers):
                        return DocumentType.EMAIL
                except:
                    pass

                # Check for JSON structure
                try:
                    content_str = content_sample.decode('utf-8', errors='ignore')
                    if (content_str.strip().startswith('{') and content_str.strip().endswith('}')) or \
                       (content_str.strip().startswith('[') and content_str.strip().endswith(']')):
                        json.loads(content_str)
                        return DocumentType.JSON
                except:
                    pass

                # Check if it's readable text
                try:
                    content_str = content_sample.decode('utf-8', errors='ignore')
                    if any(ord(c) < 128 for c in content_str):
                        # Check if it has a reasonable amount of readable characters
                        readable_chars = sum(c.isprintable() or c.isspace() for c in content_str)
                        if readable_chars / len(content_str) > 0.8:  # 80% readable characters
                            return DocumentType.TEXT
                except:
                    pass

                # Try common encodings for text files
                encodings = ['utf-8', 'ascii', 'iso-8859-1', 'utf-16', 'utf-32']
                for encoding in encodings:
                    try:
                        content_sample.decode(encoding)
                        return DocumentType.TEXT
                    except:
                        continue

            except Exception as e:
                logger.warning(f"Content-based format detection failed: {str(e)}")

            # If we reach here, try one last check for text files
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1024)  # Try to read some content
                return DocumentType.TEXT
            except:
                pass

            # If all detection methods fail, return UNKNOWN
            return DocumentType.UNKNOWN

        except Exception as e:
            logger.error(f"Format detection failed for {file_path}: {str(e)}")
            return DocumentType.UNKNOWN
        
    async def _extract_content_advanced(self, file_path: Path, format_type: str) -> str:
        """Extract text content from file with format-specific handling and advanced processing."""
        try:
            if format_type == DocumentType.PDF.value:
                return await self._extract_pdf_content(file_path)
            elif format_type == DocumentType.EMAIL.value:
                return await self._extract_email_content(file_path)
            elif format_type == DocumentType.JSON.value:
                return await self._extract_json_content(file_path)
            elif format_type == DocumentType.TEXT.value:
                return await self._extract_text_content(file_path)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
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

    async def _get_enhanced_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get enhanced file metadata with additional analysis."""
        try:
            basic_metadata = {
                "filename": file_path.name,
                "size": os.path.getsize(file_path),
                "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                "extension": file_path.suffix.lower(),
                "path": str(file_path),
                "mime_type": mimetypes.guess_type(file_path)[0]
            }

            # Add advanced metadata
            advanced_metadata = {
                "file_stats": self._get_file_stats(file_path),
                "content_info": await self._analyze_content_structure(file_path),
                "security": self._get_security_metadata(file_path)
            }

            return {**basic_metadata, **advanced_metadata}
        except Exception as e:
            logger.warning(f"Error getting enhanced metadata: {str(e)}")
            return {"filename": file_path.name, "error": str(e)}

    def _get_file_stats(self, file_path: Path) -> Dict[str, Any]:
        """Get detailed file statistics."""
        try:
            stat = os.stat(file_path)
            return {
                "size_human": self._format_size(stat.st_size),
                "permissions": oct(stat.st_mode)[-3:],
                "owner": stat.st_uid,
                "group": stat.st_gid,
                "last_access": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "is_empty": stat.st_size == 0,
                "is_executable": bool(stat.st_mode & 0o111)
            }
        except Exception as e:
            logger.warning(f"Error getting file stats: {str(e)}")
            return {}

    async def _analyze_content_structure(self, file_path: Path) -> Dict[str, Any]:
        """Analyze content structure and characteristics."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            return {
                "encoding": self._detect_encoding(content),
                "line_count": content.count(b'\n') + 1,
                "is_binary": self._is_binary(content),
                "has_bom": content.startswith(b'\xef\xbb\xbf'),
                "checksum": {
                    "md5": hashlib.md5(content).hexdigest(),
                    "sha256": hashlib.sha256(content).hexdigest()
                }
            }
        except Exception as e:
            logger.warning(f"Error analyzing content structure: {str(e)}")
            return {}

    def _get_security_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get security-related metadata."""
        try:
            return {
                "is_symlink": file_path.is_symlink(),
                "symlink_target": os.readlink(file_path) if file_path.is_symlink() else None,
                "permissions_human": self._get_human_permissions(file_path),
                "owner_name": self._get_owner_name(file_path),
                "group_name": self._get_group_name(file_path)
            }
        except Exception as e:
            logger.warning(f"Error getting security metadata: {str(e)}")
            return {}

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}PB"

    def _detect_encoding(self, content: bytes) -> str:
        """Detect content encoding."""
        encodings = ['utf-8', 'ascii', 'iso-8859-1', 'utf-16', 'utf-32']
        for encoding in encodings:
            try:
                content.decode(encoding)
                return encoding
            except:
                continue
        return "unknown"

    def _is_binary(self, content: bytes) -> bool:
        """Check if content appears to be binary."""
        textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
        return bool(content.translate(None, textchars))

    def _get_human_permissions(self, file_path: Path) -> str:
        """Get human-readable file permissions."""
        try:
            mode = os.stat(file_path).st_mode
            perms = []
            for who in ['USR', 'GRP', 'OTH']:
                for what in ['R', 'W', 'X']:
                    perms.append(what if mode & getattr(stat, f'S_I{what}{who}') else '-')
            return ''.join(perms)
        except:
            return "unknown"

    def _get_owner_name(self, file_path: Path) -> Optional[str]:
        """Get file owner name."""
        try:
            import pwd
            return pwd.getpwuid(os.stat(file_path).st_uid).pw_name
        except:
            return None

    def _get_group_name(self, file_path: Path) -> Optional[str]:
        """Get file group name."""
        try:
            import grp
            return grp.getgrgid(os.stat(file_path).st_gid).gr_name
        except:
            return None

    def _enhance_intent_result(self, result: AnalysisResult) -> Dict[str, Any]:
        """Enhance intent result with additional information."""
        # Implement intent enhancement logic here
        return {
            "type": result.primary_intent,
            "confidence": result.confidence,
            "keywords": result.keywords,
            "reasoning": result.reasoning,
            "method": result.method.value
        }

    def _generate_enhanced_routing(self, result: AnalysisResult) -> Dict[str, Any]:
        """Generate enhanced routing information."""
        try:
            suggested_department = self._get_suggested_department(result)
            
            # Determine priority based on intent and confidence
            priority = "high"
            if result.primary_intent == "FRAUD_RISK":
                priority = "critical"
            elif result.primary_intent == "COMPLAINT":
                priority = "high"
            elif result.confidence > 0.8:
                priority = "high"
            elif result.confidence > 0.5:
                priority = "medium"
            else:
                priority = "low"
                
            # Determine if human review is needed
            requires_human = (
                result.primary_intent in ["COMPLAINT", "FRAUD_RISK", "REGULATION"] or
                result.confidence < 0.6
            )
            
            return {
                "suggested_department": suggested_department,
                "priority": priority,
                "requires_human_review": requires_human,
                "confidence_based_routing": {
                    "confidence": result.confidence,
                    "confidence_level": self._get_confidence_level(result.confidence),
                    "automated_processing_recommended": result.confidence > 0.7 and not requires_human
                }
            }
        except Exception as e:
            logger.error(f"Error generating routing: {str(e)}")
            return {
                "suggested_department": "General Processing",
                "priority": "medium",
                "requires_human_review": True,
                "error": str(e)
            }

    async def _generate_advanced_fallback(self, file_path: Path, error_msg: str) -> Dict[str, Any]:
        """Generate a comprehensive fallback response with detailed error information."""
        try:
            # Get basic file information even in error case
            basic_metadata = {
                "filename": file_path.name,
                "path": str(file_path),
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
            
            # Try to get additional metadata if possible
            try:
                basic_metadata.update({
                    "size": os.path.getsize(file_path),
                    "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    "extension": file_path.suffix.lower()
                })
            except Exception as e:
                logger.warning(f"Could not get complete metadata: {str(e)}")

            # Generate comprehensive fallback response
            return {
                "format": "unknown",
                "intent": {
                    "type": "RFQ",  # Default fallback type
                    "confidence": 0.3,
                    "keywords": [],
                    "reasoning": f"Classification failed: {error_msg}",
                    "analysis_method": AnalysisMethod.FALLBACK.value
                },
                "metadata": basic_metadata,
                "analysis": {
                    "status": "error",
                    "error_type": error_msg.__class__.__name__,
                    "error_details": str(error_msg),
                    "priority_level": "high",  # High priority due to error
                    "requires_immediate_attention": True,
                    "suggested_actions": [
                        {
                            "type": "manual_review",
                            "priority": "high",
                            "action": "Review document manually",
                            "reason": "Automated classification failed"
                        },
                        {
                            "type": "error_investigation",
                            "priority": "high",
                            "action": "Investigate classification error",
                            "reason": f"Error: {error_msg}"
                        }
                    ]
                },
                "routing": {
                    "suggested_department": "Technical Support",
                    "suggested_priority": "high",
                    "requires_human_review": True,
                    "error_handling": {
                        "retry_recommended": True,
                        "error_category": "processing_error",
                        "recovery_options": [
                            "Manual processing",
                            "Retry with different format",
                            "Technical support intervention"
                        ]
                    }
                },
                "validation": {
                    "is_valid": False,
                    "validation_checks": ["Classification failed"],
                    "warnings": [str(error_msg)],
                    "recommendations": [
                        "Review document manually",
                        "Check file format and content",
                        "Contact technical support if error persists"
                    ]
                }
            }
        except Exception as e:
            # Ultimate fallback if even the advanced fallback fails
            logger.error(f"Advanced fallback generation failed: {str(e)}")
            return {
                "format": "unknown",
                "intent": {
                    "type": "RFQ",
                    "confidence": 0.1,
                    "keywords": [],
                    "reasoning": "System error - manual review required",
                    "analysis_method": "fallback"
                },
                "metadata": {"filename": file_path.name, "error": str(e)},
                "analysis": {"status": "critical_error", "requires_immediate_attention": True},
                "routing": {"requires_human_review": True}
            }

    def _generate_enhanced_summary(self, content: str, intent_analysis: Dict[str, Any]) -> str:
        """Generate a complete summary based on content and intent."""
        # Implement enhanced summary logic here
        return self._generate_summary(content, intent_analysis)

    def _generate_cache_key(self, file_path: Path) -> str:
        """Generate a unique cache key based on file content and metadata."""
        try:
            stat = os.stat(file_path)
            content_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
            return f"{file_path}:{stat.st_mtime}:{content_hash}"
        except Exception as e:
            logger.warning(f"Cache key generation failed: {str(e)}")
            return str(file_path)

    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached result is still valid."""
        if not cached_result.get('timestamp'):
            return False
            
        age = datetime.now().timestamp() - cached_result['timestamp']
        return age < self.config['cache_ttl']

    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache analysis result with timestamp."""
        self.content_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now().timestamp()
        }

    async def _detect_intent_patterns_advanced(self, content: str) -> AnalysisResult:
        """Detect intent using advanced pattern matching."""
        try:
            # Calculate pattern scores
            scores = defaultdict(float)
            matched_patterns = defaultdict(list)
            
            for intent, config in self.business_intents.items():
                for pattern, weight in config['patterns']:
                    matches = list(re.finditer(pattern, content))
                    if matches:
                        # Calculate score based on number of matches and weights
                        pattern_score = weight * len(matches)
                        scores[intent] += pattern_score
                        matched_patterns[intent].extend(
                            [match.group() for match in matches]
                        )
            
            if not scores:
                return self._generate_fallback_analysis()
                
            # Normalize scores
            total_score = sum(scores.values())
            if total_score > 0:
                scores = {k: v/total_score for k, v in scores.items()}
                
            # Get primary intent
            primary_intent = max(scores.items(), key=lambda x: x[1])[0]
            confidence = scores[primary_intent]
            
            # Generate reasoning
            pattern_matches = matched_patterns[primary_intent]
            reasoning = f"Pattern analysis found {len(pattern_matches)} matches for {primary_intent}"
            if pattern_matches:
                reasoning += f": {', '.join(pattern_matches[:3])}"
                if len(pattern_matches) > 3:
                    reasoning += f" and {len(pattern_matches)-3} more"
            
            return AnalysisResult(
                primary_intent=primary_intent,
                confidence=confidence,
                keywords=list(set(pattern_matches)),
                reasoning=reasoning,
                method=AnalysisMethod.PATTERN
            )
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {str(e)}")
            return self._generate_fallback_analysis()

    async def _detect_intent_keywords_advanced(self, content: str) -> AnalysisResult:
        """Detect intent using advanced keyword analysis."""
        try:
            # Calculate keyword scores
            scores = defaultdict(float)
            matched_keywords = defaultdict(list)
            
            content_lower = content.lower()
            for intent, config in self.business_intents.items():
                for keyword, weight in config['keywords']:
                    # Use word boundary matching for more accurate detection
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    matches = list(re.finditer(pattern, content_lower))
                    
                    if matches:
                        # Calculate score based on number of matches and weights
                        keyword_score = weight * len(matches)
                        scores[intent] += keyword_score
                        matched_keywords[intent].append(keyword)
            
            if not scores:
                return self._generate_fallback_analysis()
                
            # Normalize scores
            total_score = sum(scores.values())
            if total_score > 0:
                scores = {k: v/total_score for k, v in scores.items()}
                
            # Get primary intent
            primary_intent = max(scores.items(), key=lambda x: x[1])[0]
            confidence = scores[primary_intent]
            
            # Generate reasoning
            keywords = matched_keywords[primary_intent]
            reasoning = f"Keyword analysis found {len(keywords)} relevant keywords for {primary_intent}"
            if keywords:
                reasoning += f": {', '.join(keywords[:3])}"
                if len(keywords) > 3:
                    reasoning += f" and {len(keywords)-3} more"
            
            return AnalysisResult(
                primary_intent=primary_intent,
                confidence=confidence,
                keywords=list(set(keywords)),
                reasoning=reasoning,
                method=AnalysisMethod.KEYWORD
            )
            
        except Exception as e:
            logger.error(f"Keyword analysis failed: {str(e)}")
            return self._generate_fallback_analysis()

    def _generate_fallback_analysis(self) -> AnalysisResult:
        """Generate a fallback analysis result."""
        return AnalysisResult(
            primary_intent="RFQ",  # Default to RFQ as fallback
            confidence=0.3,
            keywords=[],
            reasoning="Fallback classification due to analysis failure",
            method=AnalysisMethod.FALLBACK
        )

    def _analyze_text_structure(self, content: str) -> Dict[str, Any]:
        """Analyze text structure characteristics."""
        try:
            lines = content.split('\n')
            words = content.split()
            
            return {
                "line_count": len(lines),
                "word_count": len(words),
                "avg_line_length": sum(len(line) for line in lines) / max(1, len(lines)),
                "avg_word_length": sum(len(word) for word in words) / max(1, len(words)),
                "unique_words": len(set(words)),
                "paragraphs": len([l for l in lines if l.strip()]),
                "has_urls": bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content))
            }
        except Exception as e:
            logger.warning(f"Error analyzing text structure: {str(e)}")
            return {
                "error": str(e),
                "line_count": len(content.split('\n')),
                "word_count": len(content.split())
            }

    def _calculate_readability(self, content: str) -> Dict[str, Any]:
        """Calculate text readability metrics."""
        try:
            words = content.split()
            sentences = re.split(r'[.!?]+', content)
            
            if not words or not sentences:
                return {"error": "Insufficient content for readability analysis"}
                
            # Simple metrics to avoid complexity
            avg_words_per_sentence = len(words) / max(1, len(sentences))
            avg_word_length = sum(len(word) for word in words) / max(1, len(words))
            
            # Simplified readability score
            simple_score = max(0, min(100, 100 - (avg_words_per_sentence * 0.5 + avg_word_length * 10)))
            
            return {
                "score": simple_score,
                "avg_words_per_sentence": avg_words_per_sentence,
                "avg_word_length": avg_word_length,
                "complexity": "high" if simple_score < 50 else "medium" if simple_score < 70 else "low"
            }
        except Exception as e:
            logger.warning(f"Error calculating readability: {str(e)}")
            return {"error": str(e)}

    def _extract_key_phrases(self, content: str) -> List[Dict[str, Any]]:
        """Extract key phrases with relevance scores."""
        try:
            # Business-related phrases to extract
            key_patterns = [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
                r'\b(?:[A-Za-z]+\s+){1,3}(?:issue|problem|request|invoice|payment|service)\b',  # Business phrases
                r'\b\d+(?:\.\d+)?%?\s+(?:increase|decrease|growth|reduction)\b',  # Metrics
                r'\b(?:urgent|immediate|critical|important)\s+(?:[a-z]+\s+){0,2}(?:action|response|attention)\b'  # Priority phrases
            ]
            
            phrases = []
            for pattern in key_patterns:
                try:
                    matches = list(re.finditer(pattern, content))
                    for match in matches[:3]:  # Limit to 3 matches per pattern
                        phrase = match.group()
                        phrases.append({
                            "text": phrase,
                            "position": match.start(),
                            "context": content[max(0, match.start()-20):min(len(content), match.end()+20)]
                        })
                except Exception as pattern_error:
                    logger.debug(f"Pattern extraction error: {str(pattern_error)}")
                    continue
                    
            return phrases[:10]  # Return at most 10 phrases
        except Exception as e:
            logger.warning(f"Error extracting key phrases: {str(e)}")
            return []

    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from content using simple pattern matching."""
        try:
            entities = {
                "emails": self._extract_emails(content),
                "urls": self._extract_urls(content),
                "dates": self._extract_dates(content),
                "amounts": self._extract_amounts(content)
            }
            
            return {k: v for k, v in entities.items() if v}
        except Exception as e:
            logger.warning(f"Error extracting entities: {str(e)}")
            return {}

    def _extract_emails(self, content: str) -> List[str]:
        """Extract email addresses from content."""
        try:
            pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            matches = re.findall(pattern, content)
            return list(set(matches))[:5]  # Return up to 5 unique emails
        except Exception:
            return []

    def _extract_urls(self, content: str) -> List[str]:
        """Extract URLs from content."""
        try:
            pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            matches = re.findall(pattern, content)
            return list(set(matches))[:5]  # Return up to 5 unique URLs
        except Exception:
            return []

    def _extract_dates(self, content: str) -> List[str]:
        """Extract dates from content."""
        try:
            # Common date formats
            date_patterns = [
                r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # DD/MM/YYYY or MM/DD/YYYY
                r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',  # DD Month YYYY
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b'  # Month DD, YYYY
            ]
            
            all_dates = []
            for pattern in date_patterns:
                all_dates.extend(re.findall(pattern, content, re.IGNORECASE))
                
            return list(set(all_dates))[:5]  # Return up to 5 unique dates
        except Exception:
            return []

    def _extract_amounts(self, content: str) -> List[str]:
        """Extract monetary amounts from content."""
        try:
            # Common money formats
            money_patterns = [
                r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?',  # $X,XXX.XX
                r'€\s*\d+(?:,\d{3})*(?:\.\d{2})?',   # €X,XXX.XX
                r'£\s*\d+(?:,\d{3})*(?:\.\d{2})?',   # £X,XXX.XX
                r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD|EUR|GBP)'  # X,XXX.XX dollars/USD/etc.
            ]
            
            all_amounts = []
            for pattern in money_patterns:
                all_amounts.extend(re.findall(pattern, content))
                
            return list(set(all_amounts))[:5]  # Return up to 5 unique amounts
        except Exception:
            return []

    def _validate_result_comprehensive(self, result: AnalysisResult) -> Dict[str, Any]:
        """Perform comprehensive validation of the analysis result."""
        try:
            # Initialize validation result
            validation = {
                "is_valid": True,
                "validation_checks": [],
                "warnings": [],
                "recommendations": []
            }
            
            # Validate confidence
            if result.confidence < 0.4:
                validation["is_valid"] = False
                validation["validation_checks"].append("confidence_check:failed")
                validation["warnings"].append(f"Low confidence score: {result.confidence:.1%}")
                validation["recommendations"].append("Manual review recommended due to low confidence")
            else:
                validation["validation_checks"].append("confidence_check:passed")
            
            # Validate intent
            if result.primary_intent not in self.business_intents:
                validation["is_valid"] = False
                validation["validation_checks"].append("intent_check:failed")
                validation["warnings"].append(f"Unknown intent type: {result.primary_intent}")
                validation["recommendations"].append("Review intent classification")
            else:
                validation["validation_checks"].append("intent_check:passed")
            
            # Validate keywords
            if not result.keywords:
                validation["validation_checks"].append("keywords_check:warning")
                validation["warnings"].append("No keywords identified")
                validation["recommendations"].append("Verify document relevance")
            else:
                validation["validation_checks"].append("keywords_check:passed")
            
            # Validate reasoning
            if not result.reasoning or len(result.reasoning) < 10:
                validation["validation_checks"].append("reasoning_check:warning")
                validation["warnings"].append("Insufficient reasoning")
                validation["recommendations"].append("Review classification basis")
            else:
                validation["validation_checks"].append("reasoning_check:passed")
            
            # Add additional validation for specific intents
            if result.primary_intent == "INVOICE":
                # For invoices, check if amounts were detected
                has_amounts = False
                if result.metadata and "entities" in result.metadata:
                    has_amounts = bool(result.metadata["entities"].get("amounts", []))
                
                if not has_amounts:
                    validation["validation_checks"].append("invoice_validation:warning")
                    validation["warnings"].append("No monetary amounts detected for INVOICE type")
                    validation["recommendations"].append("Verify if document is actually an invoice")
            
            # Add summary
            validation["summary"] = (
                "Valid result" if validation["is_valid"] 
                else "Invalid result - requires review"
            )
            
            return validation
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return {
                "is_valid": False,
                "error": str(e),
                "validation_checks": ["error_during_validation"],
                "warnings": ["Validation process failed"],
                "recommendations": ["Manual review required due to validation failure"]
            } 