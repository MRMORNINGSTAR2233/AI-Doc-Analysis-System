# AI Document Analysis System Documentation

Welcome to the documentation for the AI Document Analysis System. This documentation provides comprehensive information about the system architecture, agent logic, and how to use the application.

## Table of Contents

1. [Running the Application](./running_the_application.md)
   - Prerequisites
   - Backend Setup
   - Frontend Setup
   - Environment Variables
   - Testing the Application
   - Troubleshooting

2. [Agent Logic](./agent_logic.md)
   - Overview
   - Agent Types
   - Agent Chaining
   - Memory and Learning
   - Error Handling and Validation
   - Agent Communication

3. [Agent Flow Diagrams](./agent_flow.md)
   - Document Processing Flow
   - Agent Interaction
   - Agent Architecture

4. [Sample Outputs](./sample_outputs.md)
   - Invoice Processing
   - Email Processing
   - Processing Statistics
   - System Performance

5. [Processing Logs](../logs/processing_example.log)
   - Example of a complete document processing cycle

## System Architecture

The AI Document Analysis System is built with a modern architecture that separates the frontend and backend:

### Frontend
- **Technology**: Next.js 14, React 18, TypeScript
- **UI Components**: Tailwind CSS, Radix UI
- **State Management**: React Hooks
- **Communication**: Axios for API calls, WebSockets for real-time updates

### Backend
- **Technology**: Python 3.9, FastAPI
- **AI Processing**: LangGraph, Google Generative AI
- **Data Storage**: ChromaDB (vector database)
- **Deployment**: Docker, Docker Compose

## Getting Started

To get started with the AI Document Analysis System, follow these steps:

1. Clone the repository
2. Set up the backend (see [Running the Application](./running_the_application.md))
3. Set up the frontend (see [Running the Application](./running_the_application.md))
4. Upload a document to test the system

## Contributing

Contributions to the AI Document Analysis System are welcome! Please see the main [README.md](../README.md) file for contribution guidelines. 