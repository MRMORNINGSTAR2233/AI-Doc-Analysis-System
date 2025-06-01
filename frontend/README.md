# Frontend - AI Document Analysis System

This is the frontend application for the AI Document Analysis System, built with Next.js 14, React 18, and TypeScript.

## Features

- Modern, responsive UI with Tailwind CSS
- Real-time document processing status updates
- File upload with drag-and-drop support
- Interactive results viewer with tabs
- Dark mode support
- WebSocket integration for live updates

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI primitives
- **Forms**: React Hook Form with Zod validation
- **Icons**: Lucide React
- **HTTP Client**: Axios

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Set up environment variables (optional):
```bash
# Create .env.local file
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

3. Run the development server:
```bash
npm run dev
# or
yarn dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── src/
│   └── app/
│       ├── globals.css          # Global styles
│       ├── layout.tsx           # Root layout
│       ├── page.tsx             # Main page component
│       └── api/                 # API routes
│           ├── upload/          # File upload endpoint
│           ├── tasks/           # Task status endpoints
│           └── ws/              # WebSocket endpoints
├── public/                      # Static assets
├── package.json                 # Dependencies and scripts
├── next.config.js              # Next.js configuration
├── tailwind.config.js          # Tailwind CSS configuration
├── tsconfig.json               # TypeScript configuration
└── Dockerfile                  # Docker configuration
```

## API Integration

The frontend communicates with the backend through:

- **REST API**: For file uploads and task management
- **WebSockets**: For real-time processing updates
- **Polling**: Fallback for environments without WebSocket support

## Deployment

### Docker

Build and run with Docker:

```bash
docker build -t frontend .
docker run -p 3000:3000 frontend
```

### Docker Compose

The frontend is included in the main docker-compose.yml:

```bash
docker-compose up frontend
```

### Vercel

The application is ready for deployment on Vercel:

1. Connect your repository to Vercel
2. Set environment variables if needed
3. Deploy

## Environment Variables

- `NEXT_PUBLIC_API_URL`: Backend API URL (default: empty for relative URLs)
- `NEXT_PUBLIC_WS_URL`: WebSocket URL (default: empty for relative URLs)

## Contributing

1. Follow the existing code style and conventions
2. Use TypeScript for type safety
3. Follow the component structure in `src/app/page.tsx`
4. Test your changes thoroughly

## License

MIT
