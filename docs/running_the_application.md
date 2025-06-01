# Running the Application

This document provides instructions for running both the frontend and backend components of the AI Document Analysis System.

## Prerequisites

- Node.js 18.x or later
- Python 3.9 or later
- Docker and Docker Compose
- Git

## Backend Setup

The backend is containerized using Docker for easy deployment.

### Using Docker (Recommended)

1. Navigate to the project root:
   ```bash
   cd /path/to/Mailer
   ```

2. Build and start the backend container:
   ```bash
   docker-compose up --build
   ```

3. The backend API will be available at http://localhost:8000

### Manual Setup (Development)

1. Create a Python virtual environment:
   ```bash
   cd /path/to/Mailer/backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend server:
   ```bash
   python main.py
   ```

## Frontend Setup

The frontend is built with Next.js and can be run locally for development.

1. Navigate to the frontend directory:
   ```bash
   cd /path/to/Mailer/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. The frontend will be available at http://localhost:3000

## Environment Variables

### Backend Environment Variables

Create a `.env` file in the `backend` directory with the following variables:

```
GOOGLE_API_KEY=your_google_api_key
CHROMADB_PERSIST_DIRECTORY=/app/data/chroma
PORT=8000
ALLOWED_ORIGINS=http://localhost:3000
```

### Frontend Environment Variables

Create a `.env.local` file in the root directory with the following variables:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## Testing the Application

1. Start both the backend and frontend as described above.
2. Open http://localhost:3000 in your web browser.
3. Upload a document (PDF, email, JSON, or text file).
4. Watch the real-time processing updates.
5. View the analysis results when processing is complete.

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   
   If ports 3000 or 8000 are already in use, you can modify the port mappings in `docker-compose.yml` for the backend and use the `-p` flag for the Next.js dev server:
   ```bash
   npm run dev -- -p 3001
   ```

2. **Docker Memory Issues**
   
   If Docker runs out of memory during the build, increase the allocated memory in Docker Desktop settings.

3. **ChromaDB Initialization Errors**
   
   If ChromaDB fails to initialize, ensure the data directory exists and has proper permissions:
   ```bash
   mkdir -p backend/data/chroma
   chmod 777 backend/data/chroma
   ```

4. **WebSocket Connection Errors**
   
   If WebSocket connections fail, check that the `NEXT_PUBLIC_WS_URL` is correct and that there are no firewall or proxy issues blocking WebSocket traffic. 