import os
from pathlib import Path
import json
import logging
import sqlite3
import chromadb
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        # Initialize SQLite
        self.db_path = os.getenv("SQLITE_DB_PATH", "memory/agent_memory.db")
        self._init_sqlite()
        
        # Initialize ChromaDB
        self.chroma_path = os.getenv("CHROMA_DB_PATH", "memory/chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_memory",
            metadata={"hnsw:space": "cosine"}
        )
        
    def _init_sqlite(self):
        """Initialize SQLite database and tables."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_results (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                classification TEXT NOT NULL,
                processing_result TEXT NOT NULL,
                actions TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        
    async def store_processing_result(
        self,
        classification: Dict[str, Any],
        processing_result: Dict[str, Any],
        actions: list
    ) -> str:
        """Store processing result and return memory ID."""
        try:
            # Generate unique ID
            memory_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Store in SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO processing_results
                (id, timestamp, classification, processing_result, actions)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    timestamp,
                    json.dumps(classification),
                    json.dumps(processing_result),
                    json.dumps(actions)
                )
            )
            
            conn.commit()
            conn.close()
            
            # Store in ChromaDB for semantic search
            document_text = json.dumps({
                "classification": classification,
                "processing_result": processing_result,
                "actions": actions
            })
            
            self.collection.add(
                documents=[document_text],
                metadatas=[{
                    "timestamp": timestamp,
                    "doc_type": classification.get("format", "unknown"),
                    "memory_id": memory_id
                }],
                ids=[memory_id]
            )
            
            logger.info(f"Stored processing result with ID: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store processing result: {str(e)}")
            raise
            
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve processing result by memory ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM processing_results WHERE id = ?",
                (memory_id,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                raise ValueError(f"Memory ID not found: {memory_id}")
                
            return {
                "id": row[0],
                "timestamp": row[1],
                "classification": json.loads(row[2]),
                "processing_result": json.loads(row[3]),
                "actions": json.loads(row[4])
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {str(e)}")
            raise
            
    async def search_similar(
        self,
        query: str,
        doc_type: Optional[str] = None,
        limit: int = 5
    ) -> list:
        """Search for similar documents using ChromaDB."""
        try:
            where = {"doc_type": doc_type} if doc_type else None
            
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where
            )
            
            memory_ids = results["ids"][0]  # First query's results
            
            # Fetch full records from SQLite
            memories = []
            for memory_id in memory_ids:
                try:
                    memory = await self.get_memory(memory_id)
                    memories.append(memory)
                except ValueError:
                    continue
                    
            return memories
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {str(e)}")
            raise 