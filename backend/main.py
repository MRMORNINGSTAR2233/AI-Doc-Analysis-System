from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import aiofiles
import os
from datetime import datetime

from langgraph_core.classifier_agent import ClassifierAgent
from langgraph_core.email_agent import EmailAgent
from langgraph_core.json_agent import JSONAgent
from langgraph_core.pdf_agent import PDFAgent
from langgraph_core.action_router import ActionRouter
from memory.storage import MemoryManager

app = FastAPI(title="AI Agent System")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents and memory
memory_manager = MemoryManager()
classifier_agent = ClassifierAgent()
email_agent = EmailAgent()
json_agent = JSONAgent()
pdf_agent = PDFAgent()
action_router = ActionRouter()

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("shared/sample_inputs")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / file.filename
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
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
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        # Route actions based on processing results
        actions = await action_router.route(result)
        
        # Store in memory
        memory_id = await memory_manager.store_processing_result(
            classification=classification,
            processing_result=result,
            actions=actions
        )
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "classification": classification,
            "processing_result": result,
            "actions": actions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/{memory_id}")
async def get_memory(memory_id: str):
    try:
        memory_data = await memory_manager.get_memory(memory_id)
        return memory_data
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Memory not found: {str(e)}")

# Simulated external endpoints
@app.post("/crm/escalate")
async def crm_escalate(data: dict):
    # Simulate CRM escalation
    return {
        "status": "escalated",
        "timestamp": datetime.now().isoformat(),
        "ticket_id": "ESC-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    }

@app.post("/crm/log")
async def crm_log(data: dict):
    # Simulate CRM logging
    return {
        "status": "logged",
        "timestamp": datetime.now().isoformat(),
        "log_id": "LOG-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    }

@app.post("/risk_alert")
async def risk_alert(data: dict):
    # Simulate risk alert
    return {
        "status": "alert_created",
        "timestamp": datetime.now().isoformat(),
        "alert_id": "RISK-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 