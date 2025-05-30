# AI Agent System

A multi-agent system for processing Email, PDF, and JSON inputs with intelligent classification and automated actions.

## 🌟 Features

- **Classifier Agent**: Detects format and business intent using Gemini LLM
- **Email Agent**: Parses emails and detects tone/urgency
- **JSON Agent**: Validates webhook-like JSON inputs
- **PDF Agent**: Extracts structured data from PDFs
- **Action Router**: Triggers follow-up actions based on agent outputs
- **Modern UI**: Built with Next.js 14, Tailwind CSS, and shadcn components

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Google Cloud API Key for Gemini LLM

### Environment Setup

1. Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

### Running the Application

1. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

2. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## 📁 Project Structure

```
/frontend (Next.js UI)
  /src
    /app
    /components
/backend (FastAPI + LangGraph)
  /langgraph_core
    - classifier_agent.py
    - email_agent.py
    - json_agent.py
    - pdf_agent.py
    - action_router.py
  /memory
    - storage.py
  - main.py
/shared
  /sample_inputs
  /output_logs
```

## 🔧 Development

### Backend Development

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Development

1. Install dependencies:
   ```bash
   cd frontend
   yarn install
   ```

2. Run the development server:
   ```bash
   yarn dev
   ```

## 📝 API Documentation

### Upload Endpoint

`POST /api/upload`
- Accepts multipart form data with a file
- Supports PDF, JSON, and Email files
- Returns processing results and triggered actions

### Memory Endpoint

`GET /api/memory/{memory_id}`
- Retrieves processing results from memory storage
- Returns full classification, processing results, and actions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.