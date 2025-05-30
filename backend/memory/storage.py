from sqlalchemy import create_engine, Column, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import chromadb
import json
import uuid

Base = declarative_base()

class ProcessingRecord(Base):
    __tablename__ = 'processing_records'
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    classification = Column(JSON)
    processing_result = Column(JSON)
    actions = Column(JSON)

class MemoryManager:
    def __init__(self):
        # Initialize SQLite
        self.engine = create_engine('sqlite:///memory/agent_memory.db')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name="agent_memory",
            get_or_create=True
        )
        
    async def store_processing_result(self, classification, processing_result, actions):
        memory_id = str(uuid.uuid4())
        
        # Store in SQLite
        session = self.Session()
        record = ProcessingRecord(
            id=memory_id,
            classification=classification,
            processing_result=processing_result,
            actions=actions
        )
        session.add(record)
        session.commit()
        
        # Store in ChromaDB for semantic search
        self.collection.add(
            documents=[json.dumps({
                "classification": classification,
                "processing_result": processing_result,
                "actions": actions
            })],
            metadatas=[{"timestamp": datetime.utcnow().isoformat()}],
            ids=[memory_id]
        )
        
        return memory_id
        
    async def get_memory(self, memory_id):
        session = self.Session()
        record = session.query(ProcessingRecord).filter_by(id=memory_id).first()
        
        if not record:
            raise ValueError(f"No record found for memory_id: {memory_id}")
            
        return {
            "id": record.id,
            "timestamp": record.timestamp.isoformat(),
            "classification": record.classification,
            "processing_result": record.processing_result,
            "actions": record.actions
        }
        
    async def search_similar(self, query, n_results=5):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results 