import { NextRequest, NextResponse } from 'next/server';
import { v4 as uuidv4 } from 'uuid';
import { writeFile } from 'fs/promises';
import path from 'path';
import { mkdir } from 'fs/promises';

// Set up the uploads directory
const uploadsDir = path.join(process.cwd(), 'uploads');

export async function POST(request: NextRequest) {
  try {
    // Create uploads directory if it doesn't exist
    try {
      await mkdir(uploadsDir, { recursive: true });
    } catch (err) {
      console.log('Uploads directory already exists');
    }

    // Generate a unique task ID
    const taskId = uuidv4();
    
    // Process the form data
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }
    
    // Get file extension
    const fileExtension = file.name.split('.').pop() || '';
    
    // Create a unique filename
    const fileName = `${taskId}.${fileExtension}`;
    const filePath = path.join(uploadsDir, fileName);
    
    // Convert file to buffer and save it
    const buffer = Buffer.from(await file.arrayBuffer());
    await writeFile(filePath, buffer);
    
    // Store task info in database or filesystem
    // In a real app, you'd use a database to store task status
    
    return NextResponse.json({
      task_id: taskId,
      status: 'processing',
      message: 'File upload successful. Processing has started.',
      file_name: file.name
    });
  } catch (error) {
    console.error('Error processing upload:', error);
    return NextResponse.json(
      { error: 'Failed to process upload' },
      { status: 500 }
    );
  }
} 