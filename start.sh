#!/bin/bash

# Start the backend server
cd /app/backend
python main.py &

# Start the frontend server
cd /app/frontend
npm start 