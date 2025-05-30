from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import aiofiles
import os
from datetime import datetime
import logging
from typing import Dict, Any
from dotenv import load_dotenv

from langgraph_core.classifier_agent import ClassifierAgent
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
        # Create uploads directory if it doesn't exist
        upload_dir = Path(os.getenv("UPLOAD_DIR", "shared/uploads"))
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = upload_dir / f"{timestamp}_{file.filename}"
        
        # Save uploaded file
        try:
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file: {str(e)}"
            )
            
        # Schedule file cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_file, file_path)
            
        try:
            # Classify input
            classification = await classifier_agent.classify(file_path)
            
            # Process based on classification
            if classification["format"] == "email":
                result = await email_agent.process(file_path)
            elif classification["format"] == "json":
                result = await json_agent.process(file_path)
            elif classification["format"] == "pdf":
                result = await pdf_agent.process(file_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {classification['format']}"
                )
                
            # Route actions
            actions = await action_router.route(result)
            
            # Store in memory
            try:
                memory_id = await memory_manager.store_processing_result(
                    classification=classification,
                    processing_result=result,
                    actions=actions
                )
            except Exception as e:
                logger.error(f"Failed to store in memory: {str(e)}")
                memory_id = None
                
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "memory_id": memory_id,
                    "classification": classification,
                    "processing_result": result,
                    "actions": actions
                }
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Processing error: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )

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
    import asyncio
    
    # Ensure required directories exist
    for path in ["shared/uploads", "shared/output_logs", "memory"]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    ) 