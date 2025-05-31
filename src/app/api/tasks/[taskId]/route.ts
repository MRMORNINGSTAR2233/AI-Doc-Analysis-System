import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs/promises';

// In a real app, you'd use a database to store and retrieve task results
// For this example, we'll simulate processing and store results in memory
const taskStatuses = new Map();

export async function GET(
  request: NextRequest,
  { params }: { params: { taskId: string } }
) {
  const taskId = params.taskId;
  
  // Check if we have a status for this task
  if (!taskStatuses.has(taskId)) {
    // In a real app, you'd check your database
    // For this example, we'll simulate a new task
    taskStatuses.set(taskId, {
      status: 'processing',
      progress: {
        stage: 'initializing',
        progress: 0.1,
        details: 'Starting document analysis',
        timestamp: new Date().toISOString()
      },
      createdAt: new Date().toISOString()
    });
    
    // Simulate processing progress
    setTimeout(() => {
      taskStatuses.set(taskId, {
        status: 'processing',
        progress: {
          stage: 'format_detection',
          progress: 0.3,
          details: 'Detecting document format',
          timestamp: new Date().toISOString()
        },
        createdAt: new Date().toISOString()
      });
    }, 3000);
    
    setTimeout(() => {
      taskStatuses.set(taskId, {
        status: 'processing',
        progress: {
          stage: 'content_extraction',
          progress: 0.5,
          details: 'Extracting document content',
          timestamp: new Date().toISOString()
        },
        createdAt: new Date().toISOString()
      });
    }, 6000);
    
    setTimeout(() => {
      taskStatuses.set(taskId, {
        status: 'processing',
        progress: {
          stage: 'analysis',
          progress: 0.7,
          details: 'Analyzing document content',
          timestamp: new Date().toISOString()
        },
        createdAt: new Date().toISOString()
      });
    }, 9000);
    
    setTimeout(() => {
      taskStatuses.set(taskId, {
        status: 'completed',
        progress: {
          stage: 'completed',
          progress: 1.0,
          details: 'Analysis complete',
          timestamp: new Date().toISOString()
        },
        result: {
          task_id: taskId,
          original_filename: "document.pdf",
          timestamp: new Date().toISOString(),
          format: "pdf",
          intent: {
            type: "INVOICE",
            confidence: 0.95,
            keywords: ["invoice", "payment", "due"],
            reasoning: "This document contains invoice-specific terminology and formatting."
          },
          routing: {
            suggested_department: "Accounting",
            priority: "high"
          },
          validation: {
            is_valid: true,
            warnings: []
          },
          classification: {
            format: "pdf",
            intent: {
              type: "INVOICE",
              confidence: 0.95
            }
          },
          specialized_analysis: {
            content_analysis: {
              length: 2500,
              readability: {
                complexity: "medium"
              },
              entities: {
                emails: ["billing@example.com"],
                dates: ["2023-05-15"],
                amounts: ["$1,250.00"]
              }
            }
          }
        },
        createdAt: new Date().toISOString(),
        completedAt: new Date().toISOString()
      });
    }, 12000);
  }
  
  // Return the current status
  return NextResponse.json(taskStatuses.get(taskId) || {
    status: 'not_found',
    error: 'Task not found'
  });
} 