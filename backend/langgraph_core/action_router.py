import httpx
import logging
from typing import Dict, Any, List
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ActionRouter:
    def __init__(self):
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.client = httpx.AsyncClient()
        
    async def _execute_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action and return the result."""
        try:
            response = await self.client.post(
                f"{self.base_url}/{action['type']}",
                json={
                    "source": context.get("source", "unknown"),
                    "priority": action.get("priority", "medium"),
                    "reason": action.get("reason", "No reason provided"),
                    "metadata": context.get("metadata", {})
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "action": action["type"],
                    "response": result
                }
            else:
                logger.error(f"Action {action['type']} failed: {response.text}")
                return {
                    "status": "error",
                    "action": action["type"],
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            logger.error(f"Action execution error: {str(e)}")
            return {
                "status": "error",
                "action": action["type"],
                "error": str(e)
            }
            
    async def route(self, processing_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Route and execute actions based on processing results."""
        actions_taken = []
        
        # Extract suggested actions
        suggested_actions = processing_result.get("suggested_actions", [])
        
        # Prepare context for actions
        context = {
            "source": processing_result.get("metadata", {}).get("document_type", "unknown"),
            "metadata": {
                "analysis": processing_result.get("analysis", {}),
                "flags": processing_result.get("flags", {})
            }
        }
        
        # Execute each suggested action
        for action in suggested_actions:
            result = await self._execute_action(action, context)
            actions_taken.append({
                "type": action["type"],
                "priority": action.get("priority", "medium"),
                "reason": action.get("reason", "No reason provided"),
                "result": result
            })
            
        logger.info(f"Completed routing {len(actions_taken)} actions")
        return actions_taken
        
    async def determine_actions(self, processing_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Determine appropriate actions based on processing results from all agents.
        
        Args:
            processing_result: Combined results from classifier and specialized agents
            
        Returns:
            List of action objects with type, priority, description, and reason
        """
        actions = []
        
        # Get format and intent from classification
        document_format = processing_result.get("format", "unknown")
        intent_info = processing_result.get("intent", {})
        intent_type = intent_info.get("type", "unknown")
        confidence = intent_info.get("confidence", 0.0)
        
        # Get specialized analysis if available
        specialized_analysis = processing_result.get("specialized_analysis", {})
        
        # 1. Add actions from classification result
        if "actions" in processing_result:
            actions.extend(processing_result["actions"])
            
        # 2. Add actions from specialized analysis
        if "actions" in specialized_analysis:
            actions.extend(specialized_analysis["actions"])
            
        # 3. Generate actions based on document format
        if document_format == "email":
            # Email-specific actions
            if intent_type == "COMPLAINT":
                actions.append({
                    "type": "crm/escalate",
                    "priority": "high",
                    "description": "Escalate customer complaint",
                    "reason": "Email classified as customer complaint"
                })
            else:
                actions.append({
                    "type": "crm/log",
                    "priority": "medium",
                    "description": "Log email in CRM",
                    "reason": f"Email classified as {intent_type}"
                })
                
        elif document_format == "pdf":
            # PDF-specific actions
            if "invoice" in intent_type.lower():
                actions.append({
                    "type": "crm/log",
                    "priority": "medium",
                    "description": "Log invoice in finance system",
                    "reason": "PDF classified as invoice"
                })
                
        elif document_format == "json":
            # JSON-specific actions
            validation = specialized_analysis.get("validation", {})
            if not validation.get("is_valid", True):
                actions.append({
                    "type": "validation_error",
                    "priority": "high",
                    "description": "Report JSON validation errors",
                    "reason": f"Invalid JSON structure: {len(validation.get('errors', []))} errors"
                })
                
        # 4. Add risk-based actions
        if intent_type == "FRAUD_RISK":
            actions.append({
                "type": "risk_alert",
                "priority": "critical",
                "description": "Trigger fraud risk protocol",
                "reason": "Document classified as potential fraud risk"
            })
            
        # 5. Add confidence-based actions
        if confidence < 0.6:
            actions.append({
                "type": "validation_error",
                "priority": "medium",
                "description": "Flag for manual review",
                "reason": f"Low classification confidence: {confidence:.1%}"
            })
            
        # Deduplicate actions
        seen = set()
        unique_actions = []
        for action in actions:
            action_key = f"{action['type']}_{action.get('priority', 'medium')}"
            if action_key not in seen:
                seen.add(action_key)
                unique_actions.append(action)
                
        logger.info(f"Determined {len(unique_actions)} actions for {document_format} document")
        return unique_actions
        
    async def execute_actions(self, actions: List[Dict[str, Any]], priority_threshold: str = "high") -> List[Dict[str, Any]]:
        """Execute actions with priority at or above the threshold.
        
        Args:
            actions: List of action objects to potentially execute
            priority_threshold: Minimum priority level to execute ("low", "medium", "high", "critical")
            
        Returns:
            List of execution results
        """
        priority_levels = {
            "low": 0,
            "medium": 1,
            "high": 2,
            "critical": 3
        }
        threshold_value = priority_levels.get(priority_threshold.lower(), 1)
        
        # Filter actions by priority
        actions_to_execute = [
            action for action in actions 
            if priority_levels.get(action.get("priority", "medium").lower(), 0) >= threshold_value
        ]
        
        results = []
        
        # Prepare context
        context = {
            "source": "action_router",
            "metadata": {
                "execution_time": datetime.now().isoformat()
            }
        }
        
        # Execute filtered actions
        for action in actions_to_execute:
            result = await self._execute_action(action, context)
            results.append({
                "action": action,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
        logger.info(f"Executed {len(results)}/{len(actions)} actions (priority >= {priority_threshold})")
        return results 