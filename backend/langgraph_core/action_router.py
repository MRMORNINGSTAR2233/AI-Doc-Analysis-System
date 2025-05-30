import httpx
import logging
from typing import Dict, Any, List
import os
import json

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