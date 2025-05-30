import httpx
from typing import Dict, Any, List
import os
import json

class ActionRouter:
    def __init__(self):
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.client = httpx.AsyncClient()
        
    async def _post_crm_escalate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.client.post(
            f"{self.base_url}/crm/escalate",
            json=data
        )
        return response.json()
        
    async def _post_crm_log(self, data: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.client.post(
            f"{self.base_url}/crm/log",
            json=data
        )
        return response.json()
        
    async def _post_risk_alert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.client.post(
            f"{self.base_url}/risk_alert",
            json=data
        )
        return response.json()
        
    async def route(self, agent_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        actions = []
        
        # Handle email agent output
        if "needs_escalation" in agent_output:
            if agent_output["needs_escalation"]:
                escalation_data = {
                    "source": "email",
                    "sender": agent_output.get("sender"),
                    "subject": agent_output.get("subject"),
                    "urgency": agent_output.get("urgency"),
                    "tone": agent_output.get("tone"),
                    "request_info": agent_output.get("request_info")
                }
                result = await self._post_crm_escalate(escalation_data)
                actions.append({
                    "type": "crm_escalate",
                    "data": escalation_data,
                    "result": result
                })
            else:
                log_data = {
                    "source": "email",
                    "sender": agent_output.get("sender"),
                    "subject": agent_output.get("subject"),
                    "request_info": agent_output.get("request_info")
                }
                result = await self._post_crm_log(log_data)
                actions.append({
                    "type": "crm_log",
                    "data": log_data,
                    "result": result
                })
                
        # Handle JSON agent output
        if "risk_assessment" in agent_output and agent_output["risk_assessment"]["has_risks"]:
            risk_data = {
                "source": "json",
                "schema_type": agent_output.get("schema_type"),
                "risk_indicators": agent_output["risk_assessment"]["risk_indicators"],
                "data": agent_output.get("data")
            }
            result = await self._post_risk_alert(risk_data)
            actions.append({
                "type": "risk_alert",
                "data": risk_data,
                "result": result
            })
            
        # Handle PDF agent output
        if "needs_risk_alert" in agent_output and agent_output["needs_risk_alert"]:
            risk_data = {
                "source": "pdf",
                "violations": agent_output.get("violations", []),
                "metadata": agent_output.get("metadata", {}),
                "content": {
                    "total_amount": agent_output.get("content", {}).get("total_amount"),
                    "compliance_mentions": agent_output.get("content", {}).get("compliance_mentions")
                }
            }
            result = await self._post_risk_alert(risk_data)
            actions.append({
                "type": "risk_alert",
                "data": risk_data,
                "result": result
            })
            
        return actions 