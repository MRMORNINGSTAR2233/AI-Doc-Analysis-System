from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import aiofiles
import os
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import uuid
import tempfile
import asyncio
from pydantic import BaseModel

from langgraph_core.classifier_agent import ClassifierAgent, ProcessingStage
from langgraph_core.email_agent import EmailAgent
from langgraph_core.json_agent import JSONAgent
from langgraph_core.pdf_agent import PDFAgent
from langgraph_core.action_router import ActionRouter
from memory.storage import MemoryManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Agent System",
    description="Multi-agent system for processing various document types",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
try:
    memory_manager = MemoryManager()
    classifier_agent = ClassifierAgent()
    email_agent = EmailAgent()
    json_agent = JSONAgent()
    pdf_agent = PDFAgent()
    action_router = ActionRouter()
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

# In-memory storage for WebSocket connections and processing results
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.processing_results: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[task_id] = websocket
        
    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]
            
    async def send_update(self, task_id: str, data: Dict[str, Any]):
        if task_id in self.active_connections:
            await self.active_connections[task_id].send_json(data)
            
    def store_result(self, task_id: str, result: Dict[str, Any]):
        self.processing_results[task_id] = result
        
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.processing_results.get(task_id)

manager = ConnectionManager()

class TaskInfo(BaseModel):
    task_id: str
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

def cleanup_file(file_path: Path):
    """Clean up uploaded file after processing."""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {str(e)}")

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    try:
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Create a temporary file
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp_path = Path(temp.name)
            content = await file.read()
            temp.write(content)
        
        # Schedule file cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_file, temp_path)
            
        # Process file asynchronously
        asyncio.create_task(process_file(task_id, temp_path, file.filename))
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "File upload successful. Connect to WebSocket for progress updates."
        }
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during file upload: {str(e)}")

@app.get("/api/tasks/{task_id}")
async def get_task_result(task_id: str):
    result = manager.get_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found or processing not completed")
    return result

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await manager.connect(task_id, websocket)
    try:
        # Send initial connection confirmation
        await manager.send_update(task_id, {
            "type": "connection",
            "status": "connected",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # If result already exists, send it immediately
        if result := manager.get_result(task_id):
            await manager.send_update(task_id, {
                "type": "result",
                "status": "completed",
                "data": result,
                "timestamp": datetime.now().isoformat()
            })
        
        # Keep connection open until client disconnects
        while True:
            data = await websocket.receive_text()
            # Handle any client messages if needed
            
    except WebSocketDisconnect:
        manager.disconnect(task_id)

async def process_file(task_id: str, file_path: Path, original_filename: str):
    """Process file asynchronously and send updates via WebSocket"""
    try:
        # Progress callback function to send updates
        async def progress_callback(update: Dict[str, Any]):
            await manager.send_update(task_id, {
                "type": "progress",
                "status": "processing",
                "data": update,
                "task_id": task_id,
                "timestamp": datetime.now().isoformat()
            })
        
        # Initialize classifier agent with progress callback
        classifier = ClassifierAgent(progress_callback=progress_callback)
        
        # Send initial processing status
        await progress_callback({
            "stage": ProcessingStage.INITIALIZING.value,
            "progress": 0.0,
            "details": f"Starting processing of {original_filename}",
            "filename": original_filename
        })
        
        # STEP 1: Classify the document to determine its type
        classification_result = await classifier.classify(file_path)
        
        # Send progress update after classification
        await progress_callback({
            "stage": "specialized_processing",
            "progress": 0.5,
            "details": f"Document classified as {classification_result['format']}. Processing with specialized agent.",
            "format": classification_result['format']
        })
        
        # STEP 2: Use specialized agent based on format
        specialized_result = {}
        if classification_result['format'] == 'email':
            # Process with email agent
            specialized_result = await email_agent.process(file_path)
        elif classification_result['format'] == 'pdf':
            # Process with PDF agent
            specialized_result = await pdf_agent.process(file_path)
        elif classification_result['format'] == 'json':
            # Process with JSON agent
            specialized_result = await json_agent.process(file_path)
        else:
            # For other formats, use the classification result directly
            specialized_result = classification_result
        
        # Send progress update after specialized processing
        await progress_callback({
            "stage": "action_routing",
            "progress": 0.8,
            "details": "Determining actions based on document analysis",
            "format": classification_result['format']
        })
        
        # STEP 3: Route actions based on document analysis
        # Combine classification and specialized processing results
        combined_result = {
            **classification_result,
            "specialized_analysis": specialized_result
        }
        
        # Determine actions using the action router
        actions = await action_router.determine_actions(combined_result)
        
        # Execute high-priority automated actions
        await action_router.execute_actions(actions, priority_threshold="medium")
        
        # STEP 4: Prepare and store final result
        final_result = {
            "task_id": task_id,
            "original_filename": original_filename,
            "timestamp": datetime.now().isoformat(),
            "classification": classification_result,
            "specialized_analysis": specialized_result,
            "actions": actions,
            "format": classification_result['format'],
            "intent": classification_result['intent'],
            "routing": classification_result['routing'],
            "validation": classification_result['validation']
        }
        
        # Store the result
        manager.store_result(task_id, final_result)
        
        # Send completion notification
        await manager.send_update(task_id, {
            "type": "result",
            "status": "completed",
            "data": final_result,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        # Send error notification
        await manager.send_update(task_id, {
            "type": "error",
            "status": "failed",
            "error": str(e),
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        })
    finally:
        # Clean up temporary file
        try:
            if file_path and Path(file_path).exists():
                os.unlink(file_path)
                logger.debug(f"Successfully removed temporary file: {file_path}")
            else:
                logger.debug(f"Temporary file already removed or does not exist: {file_path}")
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")

@app.get("/api/memory/{memory_id}")
async def get_memory(memory_id: str):
    try:
        memory_data = await memory_manager.get_memory(memory_id)
        return JSONResponse(
            status_code=200,
            content=memory_data
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Memory not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to retrieve memory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve memory data"
        )

# Simulated external endpoints with proper error handling
@app.post("/crm/escalate")
async def crm_escalate(data: Dict[str, Any]):
    try:
        # Log the escalation
        logger.info(f"CRM Escalation: {json.dumps(data)}")
        
        # Simulate processing delay
        await asyncio.sleep(1)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "escalated",
                "timestamp": datetime.now().isoformat(),
                "ticket_id": f"ESC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "priority": "high",
                "assigned_to": "support_team"
            }
        )
    except Exception as e:
        logger.error(f"CRM escalation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to escalate to CRM"
        )

@app.post("/crm/log")
async def crm_log(data: Dict[str, Any]):
    try:
        # Log the entry
        logger.info(f"CRM Log: {json.dumps(data)}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "logged",
                "timestamp": datetime.now().isoformat(),
                "log_id": f"LOG-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "category": data.get("source", "general")
            }
        )
    except Exception as e:
        logger.error(f"CRM logging failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to log to CRM"
        )

@app.post("/risk_alert")
async def risk_alert(data: Dict[str, Any]):
    try:
        # Log the risk alert
        logger.info(f"Risk Alert: {json.dumps(data)}")
        
        # Simulate alert processing
        await asyncio.sleep(1)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "alert_created",
                "timestamp": datetime.now().isoformat(),
                "alert_id": f"RISK-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "severity": "high" if data.get("source") == "pdf" else "medium",
                "requires_review": True
            }
        )
    except Exception as e:
        logger.error(f"Risk alert failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create risk alert"
        )

@app.post("/validation_error")
async def validation_error(data: Dict[str, Any]):
    try:
        # Log the validation error
        logger.info(f"Validation Error: {json.dumps(data)}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "error_logged",
                "timestamp": datetime.now().isoformat(),
                "error_id": f"VAL-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "validation_errors": data.get("errors", []),
                "source": data.get("source", "unknown")
            }
        )
    except Exception as e:
        logger.error(f"Validation error logging failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to log validation error"
        )

@app.post("/manual_review")
async def manual_review(data: Dict[str, Any]):
    try:
        # Log the manual review request
        logger.info(f"Manual Review Request: {json.dumps(data)}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "queued",
                "timestamp": datetime.now().isoformat(),
                "review_id": f"REVIEW-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "priority": data.get("priority", "medium"),
                "estimated_review_time": "24h"
            }
        )
    except Exception as e:
        logger.error(f"Manual review request failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to queue manual review"
        )

@app.post("/financial_review")
async def financial_review(data: Dict[str, Any]):
    try:
        # Log the financial review request
        logger.info(f"Financial Review Request: {json.dumps(data)}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "queued",
                "timestamp": datetime.now().isoformat(),
                "review_id": f"FIN-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "priority": data.get("priority", "high"),
                "assigned_to": "finance_team"
            }
        )
    except Exception as e:
        logger.error(f"Financial review request failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to queue financial review"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "components": {
                "classifier": "ready",
                "email_agent": "ready",
                "json_agent": "ready",
                "pdf_agent": "ready",
                "action_router": "ready",
                "memory": "ready"
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Ensure required directories exist
    for path in ["shared/uploads", "shared/output_logs", "memory"]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    ) 