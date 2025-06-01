# Agent Logic Documentation

## Overview

The AI Document Analysis System uses a multi-agent approach to process documents. Each agent specializes in a specific aspect of document processing, creating a pipeline that transforms raw documents into structured data with actionable insights.

## Agent Types

### 1. Classifier Agent

**Purpose**: Determine document type, format, and intent.

**Inputs**:
- Raw document content
- Document metadata (filename, size, etc.)

**Outputs**:
- Document format (PDF, JSON, Email, Text)
- Document type classification (Invoice, Contract, Report, etc.)
- Confidence scores
- Intent analysis

**Implementation**:
```python
class ClassifierAgent:
    def __init__(self, model, vectordb):
        self.model = model
        self.vectordb = vectordb
        
    def classify(self, document):
        # Extract document features
        features = self.extract_features(document)
        
        # Compare with known document types
        similar_docs = self.vectordb.similarity_search(features)
        
        # Generate classification
        classification = self.model.predict(
            document=document,
            similar_examples=similar_docs
        )
        
        return classification
```

### 2. Extraction Agent

**Purpose**: Extract structured data from documents based on their type.

**Inputs**:
- Document content
- Classification results

**Outputs**:
- Structured data (JSON)
- Extracted entities (dates, amounts, contacts)
- Key-value pairs

**Implementation**:
```python
class ExtractionAgent:
    def __init__(self, model, schema_store):
        self.model = model
        self.schema_store = schema_store
        
    def extract(self, document, classification):
        # Get appropriate schema for document type
        schema = self.schema_store.get_schema(classification.type)
        
        # Extract structured data according to schema
        structured_data = self.model.extract(
            document=document,
            schema=schema
        )
        
        return structured_data
```

### 3. Analysis Agent

**Purpose**: Perform deep analysis on document content and extracted data.

**Inputs**:
- Document content
- Extracted structured data
- Classification results

**Outputs**:
- Content analysis (sentiment, complexity, etc.)
- Entity relationships
- Anomaly detection
- Recommendations

**Implementation**:
```python
class AnalysisAgent:
    def __init__(self, model):
        self.model = model
        
    def analyze(self, document, structured_data, classification):
        # Perform content analysis
        content_analysis = self.model.analyze_content(document)
        
        # Analyze relationships between entities
        relationships = self.model.analyze_relationships(structured_data)
        
        # Generate recommendations
        recommendations = self.model.generate_recommendations(
            document=document,
            structured_data=structured_data,
            classification=classification
        )
        
        return {
            "content_analysis": content_analysis,
            "relationships": relationships,
            "recommendations": recommendations
        }
```

### 4. Action Router

**Purpose**: Determine and execute appropriate actions based on document analysis.

**Inputs**:
- Classification results
- Extracted data
- Analysis results

**Outputs**:
- List of actions to take
- Action execution results
- Validation results

**Implementation**:
```python
class ActionRouter:
    def __init__(self, action_registry, model):
        self.action_registry = action_registry
        self.model = model
        
    def determine_actions(self, document_data):
        # Determine which actions to take
        actions = self.model.determine_actions(document_data)
        
        # Filter by priority
        priority_actions = [a for a in actions if a.priority >= "medium"]
        
        # Execute actions
        results = []
        for action in priority_actions:
            action_handler = self.action_registry.get(action.type)
            result = action_handler.execute(action.params)
            results.append(result)
            
        return results
```

## Agent Chaining

The agents work together in a chain, with each agent building on the work of the previous agents:

1. **Classifier Agent** processes the raw document
2. **Extraction Agent** uses the classification to extract structured data
3. **Analysis Agent** analyzes the document and extracted data
4. **Action Router** determines and executes actions based on all previous results

## Memory and Learning

The system uses ChromaDB to store document embeddings and examples, enabling:

1. **Similarity Search**: Find similar documents to improve classification
2. **Pattern Recognition**: Identify common patterns across documents
3. **Continuous Learning**: Improve performance over time as more documents are processed

## Error Handling and Validation

Each agent includes validation steps:

1. **Format Validation**: Ensure document format is supported
2. **Schema Validation**: Validate extracted data against expected schemas
3. **Confidence Thresholds**: Only take actions when confidence exceeds thresholds
4. **Human Review**: Flag documents for human review when confidence is low

## Agent Communication

Agents communicate through a structured message passing system:

1. **Task Manager** coordinates agent execution
2. **Progress Updates** are sent to the frontend via WebSockets
3. **Results** are stored in a standardized format for consistency 