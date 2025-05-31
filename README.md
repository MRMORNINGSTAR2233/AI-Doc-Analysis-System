# Mailer - AI Document Analysis System

A Next.js application for processing and analyzing documents using AI.

## Features

- Document upload and processing
- AI-powered document analysis
- Classification of document types
- Entity extraction
- Intent recognition

## Tech Stack

- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS
- **Backend**: Next.js API Routes
- **Styling**: Tailwind CSS, Radix UI
- **Form Handling**: react-hook-form, zod

## Deployment on Vercel

### Prerequisites

- A [Vercel](https://vercel.com) account
- A GitHub, GitLab, or Bitbucket repository with your code

### Deployment Steps

1. **Push your code to a Git repository**

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repository-url>
git push -u origin main
```

2. **Deploy to Vercel**

- Go to [Vercel Dashboard](https://vercel.com/dashboard)
- Click "Add New" → "Project"
- Import your Git repository
- Configure your project:
  - Framework Preset: Next.js
  - Root Directory: ./
  - Build Command: `npm run build`
  - Output Directory: .next
- Click "Deploy"

3. **Environment Variables**

No environment variables are required for basic functionality.

## Local Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Project Structure

- `src/app/` - Next.js App Router pages and components
- `src/app/api/` - API routes
- `uploads/` - Temporary storage for uploaded files

## License

MIT
