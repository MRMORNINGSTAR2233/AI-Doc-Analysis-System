import os
import google.generativeai as genai
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Optional
import pypdf
import re
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class PDFAgent:
    def __init__(self):
        # Initialize Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key or api_key == "your_google_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required. "
                "Please set a valid Google Generative AI API key in your .env file."
            )
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("PDF Agent: Successfully initialized Google Generative AI model")
        except Exception as e:
            logger.error(f"PDF Agent: Failed to initialize Gemini model: {str(e)}")
            raise RuntimeError(f"Could not initialize Google Generative AI: {str(e)}")
        
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
        
        # More specific invoice item patterns that avoid false matches
        patterns = [
            # Pattern 1: $amount - description
            r"\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*[-–]\s*(.+?)(?=\$|\n|$)",
            # Pattern 2: quantity x item @ $price
            r"(\d+)\s*x\s*(.+?)\s*@\s*\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)",
            # Pattern 3: description $amount (more restrictive)
            r"^(.+?)\s+\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)(?:\s*USD)?\s*$",
            # Pattern 4: amount USD - description
            r"(\d+(?:,\d{3})*(?:\.\d{2})?)\s*USD\s*[-–]\s*(.+?)(?=\n|$)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                try:
                    groups = match.groups()
                    
                    if len(groups) == 2:
                        # Check which group contains the amount
                        group1, group2 = groups
                        
                        # Try to determine which is amount and which is description
                        if self._is_valid_amount(group1):
                            amount_str = group1
                            description = group2.strip()
                        elif self._is_valid_amount(group2):
                            amount_str = group2
                            description = group1.strip()
                        else:
                            continue  # Skip if neither group is a valid amount
                        
                        # Convert amount to float
                        amount = float(amount_str.replace(',', ''))
                        
                        # Validate description (skip if it looks like metadata)
                        if self._is_valid_description(description):
                            items.append({
                                "amount": amount,
                                "description": description
                            })
                            
                    elif len(groups) == 3:  # Quantity, item, and unit price
                        quantity_str, description, unit_price_str = groups
                        
                        # Validate all components
                        if (self._is_valid_quantity(quantity_str) and 
                            self._is_valid_amount(unit_price_str) and 
                            self._is_valid_description(description)):
                            
                            quantity = int(quantity_str)
                            unit_price = float(unit_price_str.replace(',', ''))
                            
                            items.append({
                                "quantity": quantity,
                                "description": description.strip(),
                                "unit_price": unit_price,
                                "amount": quantity * unit_price
                            })
                            
                except (ValueError, AttributeError) as e:
                    # Log the error but continue processing other matches
                    logger.debug(f"Skipping invalid invoice item match: {match.group()} - {str(e)}")
                    continue
                    
        return items
    
    def _is_valid_amount(self, text: str) -> bool:
        """Check if text represents a valid monetary amount."""
        if not text or not isinstance(text, str):
            return False
        
        # Clean the text
        cleaned = text.strip().replace(',', '').replace('$', '').replace('USD', '').strip()
        
        # Check if it's a valid number
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    
    def _is_valid_quantity(self, text: str) -> bool:
        """Check if text represents a valid quantity."""
        if not text or not isinstance(text, str):
            return False
        
        try:
            qty = int(text.strip())
            return qty > 0 and qty < 10000  # Reasonable quantity limits
        except ValueError:
            return False
    
    def _is_valid_description(self, text: str) -> bool:
        """Check if text represents a valid item description."""
        if not text or not isinstance(text, str):
            return False
        
        text = text.strip()
        
        # Skip if too short or contains invoice metadata
        if len(text) < 3:
            return False
        
        # Skip common invoice metadata patterns
        invalid_patterns = [
            r'invoice\s+number',
            r'invoice\s+date',
            r'due\s+date',
            r'bill\s+to',
            r'ship\s+to',
            r'payment\s+terms',
            r'total\s+amount',
            r'subtotal',
            r'tax\s+rate',
            r'grand\s+total'
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        return True
        
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
        
        Document text:
        {text[:3000]}...
        """
        
        try:
            response = await self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.3,
                    'top_p': 0.8,
                    'top_k': 40
                }
            )
            result = json.loads(response.text)
            return self._validate_analysis_result(result)
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            return self._generate_fallback_analysis(text)

    def _validate_analysis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean analysis result."""
        # Ensure valid document type
        valid_types = ["invoice", "contract", "report", "policy", "other"]
        if not isinstance(result.get("document_type"), str) or result["document_type"].lower() not in valid_types:
            # Try to infer document type from content before defaulting to other
            inferred_type = self._infer_document_type(result)
            result["document_type"] = inferred_type if inferred_type else "other"

        # Ensure valid lists with proper validation
        if not isinstance(result.get("compliance_requirements"), list):
            result["compliance_requirements"] = []
        else:
            # Clean and validate compliance requirements
            result["compliance_requirements"] = [
                str(req) for req in result["compliance_requirements"] 
                if isinstance(req, (str, int, float)) and str(req).strip()
            ]

        # Validate key findings with type checking
        if not isinstance(result.get("key_findings"), dict):
            result["key_findings"] = {
                "monetary_values": [],
                "dates": [],
                "entities": [],
                "critical_terms": []
            }
        else:
            # Ensure all key_findings sub-fields are lists
            for key in ["monetary_values", "dates", "entities", "critical_terms"]:
                if not isinstance(result["key_findings"].get(key), list):
                    result["key_findings"][key] = []
                else:
                    # Clean and validate list items
                    result["key_findings"][key] = [
                        str(item) for item in result["key_findings"][key]
                        if isinstance(item, (str, int, float)) and str(item).strip()
                    ]

        # Validate risk assessment with detailed checks
        if not isinstance(result.get("risk_assessment"), dict):
            result["risk_assessment"] = {
                "level": "medium",
                "factors": [],
                "requires_review": True,
                "confidence": 0.5,
                "explanation": "Default risk assessment due to validation failure"
            }
        else:
            risk = result["risk_assessment"]
            # Validate risk level
            if not isinstance(risk.get("level"), str) or risk["level"].lower() not in ["high", "medium", "low"]:
                risk["level"] = "medium"
            
            # Validate factors list
            if not isinstance(risk.get("factors"), list):
                risk["factors"] = []
            else:
                risk["factors"] = [str(f) for f in risk["factors"] if f and isinstance(f, (str, int, float))]
            
            # Ensure boolean requires_review
            risk["requires_review"] = bool(risk.get("requires_review", True))
            
            # Add confidence score if missing
            if "confidence" not in risk or not isinstance(risk["confidence"], (int, float)):
                risk["confidence"] = 0.5
            risk["confidence"] = max(0.0, min(1.0, float(risk["confidence"])))
            
            # Add explanation if missing
            if not isinstance(risk.get("explanation"), str):
                risk["explanation"] = f"Risk level {risk['level']} determined with {risk['confidence']:.1%} confidence"

        # Validate and enhance suggested actions
        result["suggested_actions"] = self._enhance_suggested_actions(result)

        # Add confidence metrics
        result["analysis_confidence"] = {
            "overall": self._calculate_overall_confidence(result),
            "document_type": self._calculate_type_confidence(result),
            "risk_assessment": result["risk_assessment"].get("confidence", 0.5),
            "explanation": self._generate_confidence_explanation(result)
        }

        return result

    def _infer_document_type(self, result: Dict[str, Any]) -> Optional[str]:
        """Infer document type from analysis results."""
        # Check key findings for type indicators
        key_findings = result.get("key_findings", {})
        
        # Check for invoice indicators
        if (
            key_findings.get("monetary_values")
            or "invoice" in str(result).lower()
            or "payment" in str(result).lower()
        ):
            return "invoice"
            
        # Check for contract indicators
        if (
            "agreement" in str(result).lower()
            or "contract" in str(result).lower()
            or "terms" in str(result).lower()
        ):
            return "contract"
            
        # Check for report indicators
        if (
            "report" in str(result).lower()
            or "analysis" in str(result).lower()
            or "summary" in str(result).lower()
        ):
            return "report"
            
        # Check for policy indicators
        if (
            "policy" in str(result).lower()
            or "regulation" in str(result).lower()
            or "compliance" in str(result).lower()
        ):
            return "policy"
            
        return None

    def _calculate_overall_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        confidence_scores = [
            self._calculate_type_confidence(result),
            result.get("risk_assessment", {}).get("confidence", 0.5),
            0.7 if result.get("key_findings", {}).get("monetary_values") else 0.5,
            0.7 if result.get("compliance_requirements") else 0.5
        ]
        return sum(confidence_scores) / len(confidence_scores)

    def _calculate_type_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence in document type classification."""
        doc_type = result.get("document_type", "other")
        
        if doc_type == "other":
            return 0.3
            
        # Check for type-specific indicators
        indicators = {
            "invoice": ["invoice", "payment", "amount", "bill", "cost"],
            "contract": ["agreement", "terms", "parties", "contract"],
            "report": ["report", "analysis", "findings", "summary"],
            "policy": ["policy", "regulation", "compliance", "rules"]
        }
        
        if doc_type in indicators:
            matches = sum(1 for indicator in indicators[doc_type] 
                        if indicator in str(result).lower())
            return min(0.9, 0.4 + (matches * 0.1))
            
        return 0.5

    def _generate_confidence_explanation(self, result: Dict[str, Any]) -> str:
        """Generate detailed explanation of confidence levels."""
        overall_conf = self._calculate_overall_confidence(result)
        type_conf = self._calculate_type_confidence(result)
        
        explanation = []
        
        # Document type confidence
        explanation.append(
            f"Document type '{result.get('document_type', 'unknown')}' "
            f"determined with {type_conf:.1%} confidence"
        )
        
        # Risk assessment confidence
        risk_conf = result.get("risk_assessment", {}).get("confidence", 0.5)
        explanation.append(
            f"Risk assessment confidence: {risk_conf:.1%}"
        )
        
        # Key findings confidence
        if result.get("key_findings", {}).get("monetary_values"):
            explanation.append("High confidence in monetary value detection")
        
        # Overall assessment
        explanation.append(
            f"Overall analysis confidence: {overall_conf:.1%}"
        )
        
        return " | ".join(explanation)

    def _enhance_suggested_actions(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate and enhance suggested actions based on analysis results."""
        actions = []
        
        # Get existing actions or initialize empty list
        existing_actions = analysis_result.get("suggested_actions", [])
        if isinstance(existing_actions, list):
            actions.extend(existing_actions)

        # Add actions based on document type
        doc_type = analysis_result.get("document_type", "other")
        if doc_type == "invoice":
            actions.append({
                "type": "invoice_processing",
                "priority": "high",
                "description": "Process invoice for payment",
                "reason": "Document identified as invoice"
            })
        elif doc_type == "contract":
            actions.append({
                "type": "legal_review",
                "priority": "high",
                "description": "Review contract terms",
                "reason": "Document identified as contract"
            })

        # Add actions based on risk assessment
        risk_assessment = analysis_result.get("risk_assessment", {})
        if risk_assessment.get("level") == "high":
            actions.append({
                "type": "risk_review",
                "priority": "high",
                "description": "Conduct detailed risk assessment",
                "reason": "High risk level detected"
            })

        # Add actions based on compliance requirements
        compliance_reqs = analysis_result.get("compliance_requirements", [])
        if compliance_reqs:
            actions.append({
                "type": "compliance_review",
                "priority": "high",
                "description": "Review compliance requirements",
                "reason": f"Compliance requirements detected: {', '.join(compliance_reqs)}"
            })

        # Add actions based on key findings
        key_findings = analysis_result.get("key_findings", {})
        if key_findings.get("monetary_values"):
            actions.append({
                "type": "financial_review",
                "priority": "high",
                "description": "Review financial implications",
                "reason": "Monetary values detected in document"
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

    async def process(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF file and return analysis."""
        processing_steps = []
        
        try:
            # Extract text
            logger.info(f"Starting PDF processing for {file_path.name}")
            text = self._extract_text(file_path)
            processing_steps.append({"step": "text_extraction", "status": "success"})
            
            # Extract invoice items if present (with error handling)
            invoice_items = []
            try:
                invoice_items = self._extract_invoice_items(text)
                processing_steps.append({
                    "step": "invoice_detection", 
                    "status": "success", 
                    "items_found": len(invoice_items)
                })
            except Exception as e:
                logger.warning(f"Invoice item extraction failed: {str(e)}")
                processing_steps.append({
                    "step": "invoice_detection", 
                    "status": "failed", 
                    "error": str(e)
                })
            
            # Check for compliance keywords
            compliance_findings = {}
            try:
                compliance_findings = self._check_compliance_keywords(text)
                processing_steps.append({
                    "step": "compliance_check", 
                    "status": "success", 
                    "regulations_found": len(compliance_findings)
                })
            except Exception as e:
                logger.warning(f"Compliance check failed: {str(e)}")
                processing_steps.append({
                    "step": "compliance_check", 
                    "status": "failed", 
                    "error": str(e)
                })
            
            # Analyze content
            content_analysis = {}
            try:
                content_analysis = await self._analyze_content(text)
                processing_steps.append({
                    "step": "content_analysis", 
                    "status": "success", 
                    "risk_level": content_analysis.get("risk_assessment", {}).get("level", "unknown")
                })
            except Exception as e:
                logger.warning(f"Content analysis failed: {str(e)}")
                content_analysis = self._generate_fallback_analysis(text)
                processing_steps.append({
                    "step": "content_analysis", 
                    "status": "fallback", 
                    "error": str(e)
                })
            
            # Calculate total value for invoices (with safety checks)
            total_value = 0
            if invoice_items:
                try:
                    total_value = sum(item.get("amount", 0) for item in invoice_items if isinstance(item.get("amount"), (int, float)))
                except Exception as e:
                    logger.warning(f"Total value calculation failed: {str(e)}")
                    total_value = 0
            
            # Generate actions
            actions = []
            try:
                actions = self._generate_actions(
                    content_analysis,
                    total_value,
                    compliance_findings,
                    invoice_items
                )
                processing_steps.append({
                    "step": "action_generation", 
                    "status": "success", 
                    "actions_generated": len(actions)
                })
            except Exception as e:
                logger.warning(f"Action generation failed: {str(e)}")
                processing_steps.append({
                    "step": "action_generation", 
                    "status": "failed", 
                    "error": str(e)
                })
            
            # Prepare result with safe access to all fields
            result = {
                "metadata": {
                    "file_name": file_path.name,
                    "document_type": content_analysis.get("document_type", "unknown"),
                    "page_count": self._safe_get_page_count(file_path)
                },
                "content": {
                    "text_preview": text[:1000] + "..." if len(text) > 1000 else text,
                    "invoice_items": invoice_items if invoice_items else None,
                    "total_value": total_value if invoice_items else None
                },
                "analysis": {
                    "compliance": {
                        "regulations_mentioned": list(compliance_findings.keys()) if compliance_findings else [],
                        "keyword_matches": compliance_findings,
                        "requirements": content_analysis.get("compliance_requirements", [])
                    },
                    "key_findings": content_analysis.get("key_findings", {}),
                    "risk_assessment": content_analysis.get("risk_assessment", {
                        "level": "medium",
                        "factors": [],
                        "requires_review": True
                    })
                },
                "flags": {
                    "high_value": total_value >= 10000 if total_value else False,
                    "requires_review": content_analysis.get("risk_assessment", {}).get("requires_review", True),
                    "has_compliance_terms": len(compliance_findings) > 0
                },
                "actions": actions,
                "suggested_actions": content_analysis.get("suggested_actions", [])
            }
            
            # Add memory trace
            result["memory_trace"] = {
                "processing_steps": processing_steps,
                "decision_factors": {
                    "high_value": result["flags"]["high_value"],
                    "compliance_terms": result["flags"]["has_compliance_terms"],
                    "high_risk": content_analysis.get("risk_assessment", {}).get("level") == "high"
                }
            }
            
            logger.info(f"PDF processing complete for {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            # Return a basic error result instead of raising
            return self._generate_error_result(file_path, str(e), processing_steps)

    def _safe_get_page_count(self, file_path: Path) -> int:
        """Safely get page count from PDF."""
        try:
            reader = pypdf.PdfReader(file_path)
            return len(reader.pages)
        except Exception as e:
            logger.warning(f"Could not get page count: {str(e)}")
            return 0

    def _generate_error_result(self, file_path: Path, error_message: str, processing_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a fallback result when processing fails."""
        return {
            "metadata": {
                "file_name": file_path.name,
                "document_type": "unknown",
                "page_count": 0,
                "processing_error": error_message
            },
            "content": {
                "text_preview": "Error: Could not extract content",
                "invoice_items": None,
                "total_value": None
            },
            "analysis": {
                "compliance": {
                    "regulations_mentioned": [],
                    "keyword_matches": {},
                    "requirements": []
                },
                "key_findings": {},
                "risk_assessment": {
                    "level": "high",
                    "factors": ["processing_error"],
                    "requires_review": True
                }
            },
            "flags": {
                "high_value": False,
                "requires_review": True,
                "has_compliance_terms": False,
                "processing_error": True
            },
            "actions": [{
                "type": "manual_review",
                "priority": "high",
                "description": "Manual review required due to processing error",
                "reason": f"Automated processing failed: {error_message}"
            }],
            "suggested_actions": [],
            "memory_trace": {
                "processing_steps": processing_steps + [{
                    "step": "error_handling",
                    "status": "completed",
                    "error": error_message
                }],
                "decision_factors": {
                    "processing_error": True
                }
            }
        }

    def _generate_actions(
        self,
        content_analysis: Dict[str, Any],
        total_value: float,
        compliance_findings: Dict[str, List[str]],
        invoice_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate actions based on document analysis."""
        actions = []
        
        # Add actions from content analysis
        if "suggested_actions" in content_analysis:
            actions.extend(content_analysis["suggested_actions"])
        
        # Add high-value actions
        if total_value >= 10000:
            actions.append({
                "type": "financial_review",
                "priority": "high",
                "description": "Review high-value transaction",
                "reason": f"Document contains high value amount: ${total_value:,.2f}"
            })
        
        # Add compliance-related actions
        if compliance_findings:
            for regulation, keywords in compliance_findings.items():
                actions.append({
                    "type": "compliance_review",
                    "priority": "high",
                    "description": f"Review {regulation} compliance",
                    "reason": f"Found {regulation} related terms: {', '.join(keywords)}"
                })
        
        # Add invoice-specific actions
        if invoice_items:
            actions.append({
                "type": "invoice_processing",
                "priority": "high" if total_value >= 10000 else "medium",
                "description": "Process invoice for payment",
                "reason": f"Invoice with {len(invoice_items)} items, total value: ${total_value:,.2f}"
            })
        
        # Add risk-based actions
        risk_level = content_analysis.get("risk_assessment", {}).get("level", "medium")
        if risk_level == "high":
            actions.append({
                "type": "risk_mitigation",
                "priority": "high",
                "description": "Review and mitigate identified risks",
                "reason": "High risk level detected in document"
            })
        
        return actions

    async def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text content from PDF."""
        try:
            reader = pypdf.PdfReader(file_path)
            content = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content.append(text)
                    
            return "\n\n".join(content)
            
        except Exception as e:
            logger.error(f"PDF content extraction error: {str(e)}")
            return ""
            
    async def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract PDF metadata."""
        try:
            reader = pypdf.PdfReader(file_path)
            metadata = reader.metadata
            
            result = {
                "title": metadata.get("/Title", ""),
                "author": metadata.get("/Author", ""),
                "subject": metadata.get("/Subject", ""),
                "creator": metadata.get("/Creator", ""),
                "producer": metadata.get("/Producer", ""),
                "creation_date": metadata.get("/CreationDate", ""),
                "modification_date": metadata.get("/ModDate", ""),
                "pages": len(reader.pages)
            }
            
            # Clean up dates
            for date_field in ["creation_date", "modification_date"]:
                if result[date_field] and isinstance(result[date_field], str):
                    try:
                        # Convert PDF date format to ISO
                        date_str = result[date_field].replace("D:", "").split("+")[0]
                        date_obj = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                        result[date_field] = date_obj.isoformat()
                    except:
                        pass
                        
            return result
            
        except Exception as e:
            logger.error(f"PDF metadata extraction error: {str(e)}")
            return {}
            
    async def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from content."""
        entities = {
            "dates": self._extract_dates(content),
            "monetary_values": self._extract_monetary_values(content),
            "email_addresses": self._extract_emails(content),
            "phone_numbers": self._extract_phone_numbers(content),
            "urls": self._extract_urls(content)
        }
        return entities
        
    def _extract_dates(self, content: str) -> List[str]:
        """Extract dates using regex."""
        pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}'
        return list(set(re.findall(pattern, content, re.IGNORECASE)))
        
    def _extract_monetary_values(self, content: str) -> List[str]:
        """Extract monetary values using regex."""
        pattern = r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)'
        return list(set(re.findall(pattern, content)))
        
    def _extract_emails(self, content: str) -> List[str]:
        """Extract email addresses using regex."""
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return list(set(re.findall(pattern, content)))
        
    def _extract_phone_numbers(self, content: str) -> List[str]:
        """Extract phone numbers using regex."""
        pattern = r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        return list(set(re.findall(pattern, content)))
        
    def _extract_urls(self, content: str) -> List[str]:
        """Extract URLs using regex."""
        pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        return list(set(re.findall(pattern, content)))
        
    async def _generate_summary(self, content: str) -> Dict[str, Any]:
        """Generate document summary using Gemini."""
        prompt = f"""Generate a concise summary of this PDF document content.
        Include:
        - Main points
        - Key findings
        - Important details
        - Conclusions or recommendations

        Provide your summary in this exact JSON format:
        {{
            "brief_summary": "2-3 sentence overview",
            "main_points": ["point1", "point2"],
            "key_findings": ["finding1", "finding2"],
            "conclusions": ["conclusion1", "conclusion2"],
            "confidence_level": <float between 0.0-1.0>
        }}

        Content to summarize:
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
            
            result = json.loads(response.text.strip())
            return self._validate_summary(result)
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return self._generate_fallback_summary()
            
    def _validate_content_analysis(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean content analysis result."""
        valid_doc_types = ["report", "contract", "invoice", "presentation", "other"]
        if not isinstance(result.get("document_type"), str) or result["document_type"].lower() not in valid_doc_types:
            result["document_type"] = "other"
            
        if not isinstance(result.get("key_topics"), list):
            result["key_topics"] = []
            
        if not isinstance(result.get("important_sections"), list):
            result["important_sections"] = []
            
        valid_complexity = ["high", "medium", "low"]
        if not isinstance(result.get("complexity_level"), str) or result["complexity_level"].lower() not in valid_complexity:
            result["complexity_level"] = "medium"
            
        if not isinstance(result.get("target_audience"), list):
            result["target_audience"] = ["general"]
            
        if not isinstance(result.get("action_items"), list):
            result["action_items"] = []
            
        try:
            result["confidence_level"] = float(result.get("confidence_level", 0.5))
            result["confidence_level"] = max(0.0, min(1.0, result["confidence_level"]))
        except (TypeError, ValueError):
            result["confidence_level"] = 0.5
            
        return result
        
    def _validate_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean summary result."""
        if not isinstance(result.get("brief_summary"), str):
            result["brief_summary"] = "Summary not available"
            
        for key in ["main_points", "key_findings", "conclusions"]:
            if not isinstance(result.get(key), list):
                result[key] = []
                
        try:
            result["confidence_level"] = float(result.get("confidence_level", 0.5))
            result["confidence_level"] = max(0.0, min(1.0, result["confidence_level"]))
        except (TypeError, ValueError):
            result["confidence_level"] = 0.5
            
        return result
        
    def _generate_fallback_analysis(self, content: str) -> Dict[str, Any]:
        """Generate fallback content analysis when AI analysis fails."""
        return {
            "document_type": "other",
            "compliance_requirements": [],
            "key_findings": {
                "monetary_values": [],
                "dates": [],
                "entities": [],
                "critical_terms": []
            },
            "risk_assessment": {
                "level": "medium",
                "factors": ["analysis_failed"],
                "requires_review": True,
                "confidence": 0.3,
                "explanation": "Automated analysis failed, manual review recommended"
            },
            "suggested_actions": [
                {
                    "type": "manual_review",
                    "priority": "high",
                    "description": "Conduct manual document review",
                    "reason": "Automated analysis failed"
                }
            ],
            "analysis_confidence": {
                "overall": 0.3,
                "document_type": 0.3,
                "risk_assessment": 0.3,
                "explanation": "Fallback analysis due to processing failure"
            }
        }
        
    def _generate_fallback_summary(self) -> Dict[str, Any]:
        """Generate fallback summary."""
        return {
            "brief_summary": "Summary generation failed",
            "main_points": ["Manual review recommended"],
            "key_findings": [],
            "conclusions": ["Unable to generate automatic summary"],
            "confidence_level": 0.3
        } 