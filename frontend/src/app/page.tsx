'use client'

import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import * as z from 'zod'
import axios from 'axios'
import * as Progress from '@radix-ui/react-progress'
import * as Separator from '@radix-ui/react-separator'
import * as Tabs from '@radix-ui/react-tabs'
import { Upload, AlertTriangle, FileText, Mail, FileJson, CheckCircle, XCircle, AlertCircle } from 'lucide-react'

const formSchema = z.object({
  file: z.instanceof(File)
})

export default function Home() {
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })
  
  const onSubmit = async (data: z.infer<typeof formSchema>) => {
    try {
      setProcessing(true)
      setError(null)
      setUploadProgress(0)
      
      const formData = new FormData()
      formData.append('file', data.file)
      
      const response = await axios.post('http://localhost:8000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const progress = progressEvent.total
            ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
            : 0
          setUploadProgress(progress)
        }
      })
      
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'An error occurred')
    } finally {
      setProcessing(false)
    }
  }
  
  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'pdf':
        return <FileText className="h-5 w-5" />
      case 'email':
        return <Mail className="h-5 w-5" />
      case 'json':
        return <FileJson className="h-5 w-5" />
      default:
        return <FileText className="h-5 w-5" />
    }
  }
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'error':
        return <XCircle className="h-5 w-5 text-red-500" />
      case 'warning':
        return <AlertCircle className="h-5 w-5 text-yellow-500" />
      default:
        return null
    }
  }
  
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <h1 className="text-2xl font-bold text-gray-900">AI Agent System</h1>
          </div>
        </div>
      </nav>
      
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="grid gap-8 md:grid-cols-2">
            {/* Upload Section */}
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Upload Document</h2>
                
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 transition-colors hover:border-blue-400">
                    <input
                      type="file"
                      onChange={(e) => {
                        if (e.target.files?.[0]) {
                          form.setValue('file', e.target.files[0])
                        }
                      }}
                      className="hidden"
                      id="file-upload"
                      accept=".pdf,.json,.eml,.txt"
                    />
                    <label
                      htmlFor="file-upload"
                      className="cursor-pointer flex flex-col items-center"
                    >
                      <Upload className="h-12 w-12 text-gray-400 mb-3" />
                      <span className="text-sm font-medium text-gray-700">
                        Drop your file here or click to upload
                      </span>
                      <span className="mt-1 text-xs text-gray-500">
                        Supports PDF, JSON, and Email files
                      </span>
                    </label>
                  </div>
                  
                  {form.watch('file') && (
                    <div className="flex items-center space-x-2 text-sm text-gray-600">
                      <FileText className="h-4 w-4" />
                      <span>{form.watch('file').name}</span>
                    </div>
                  )}
                  
                  {processing && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-700">Uploading...</span>
                        <span className="text-gray-500">{uploadProgress}%</span>
                      </div>
                      <Progress.Root
                        className="h-2 overflow-hidden bg-gray-200 rounded-full"
                        value={uploadProgress}
                      >
                        <Progress.Indicator
                          className="h-full bg-blue-500 transition-all duration-300 ease-in-out"
                          style={{ width: `${uploadProgress}%` }}
                        />
                      </Progress.Root>
                    </div>
                  )}
                  
                  <button
                    type="submit"
                    disabled={processing || !form.watch('file')}
                    className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {processing ? 'Processing...' : 'Process Document'}
                  </button>
                </form>
                
                {error && (
                  <div className="mt-4 p-4 bg-red-50 rounded-md">
                    <div className="flex">
                      <AlertTriangle className="h-5 w-5 text-red-400" />
                      <div className="ml-3">
                        <h3 className="text-sm font-medium text-red-800">Error</h3>
                        <p className="mt-1 text-sm text-red-700">{error}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            {/* Results Section */}
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Results</h2>
                
                {result ? (
                  <Tabs.Root defaultValue="classification" className="space-y-6">
                    <Tabs.List className="flex space-x-4 border-b border-gray-200">
                      <Tabs.Trigger
                        value="classification"
                        className="px-3 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent data-[state=active]:border-blue-500 data-[state=active]:text-blue-600"
                      >
                        Classification
                      </Tabs.Trigger>
                      <Tabs.Trigger
                        value="processing"
                        className="px-3 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent data-[state=active]:border-blue-500 data-[state=active]:text-blue-600"
                      >
                        Processing
                      </Tabs.Trigger>
                      <Tabs.Trigger
                        value="actions"
                        className="px-3 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent data-[state=active]:border-blue-500 data-[state=active]:text-blue-600"
                      >
                        Actions
                      </Tabs.Trigger>
                    </Tabs.List>
                    
                    <Tabs.Content value="classification" className="space-y-4">
                      <div className="rounded-md bg-gray-50 p-4">
                        <div className="flex items-center space-x-2">
                          {getFormatIcon(result.classification.format)}
                          <span className="text-sm font-medium text-gray-900">
                            Format: {result.classification.format}
                          </span>
                        </div>
                        <Separator.Root className="my-4 h-px bg-gray-200" />
                        <div className="text-sm text-gray-600">
                          <p>Intent: {result.classification.intent}</p>
                          <p className="mt-2">Memory ID: {result.memory_id}</p>
                        </div>
                      </div>
                    </Tabs.Content>
                    
                    <Tabs.Content value="processing" className="space-y-4">
                      <pre className="bg-gray-50 rounded-md p-4 text-sm text-gray-800 overflow-auto max-h-96">
                        {JSON.stringify(result.processing_result, null, 2)}
                      </pre>
                    </Tabs.Content>
                    
                    <Tabs.Content value="actions" className="space-y-4">
                      {result.actions.map((action: any, index: number) => (
                        <div
                          key={index}
                          className="flex items-center space-x-3 p-4 bg-blue-50 rounded-md"
                        >
                          {getStatusIcon(action.result.status)}
                          <div>
                            <p className="text-sm font-medium text-blue-900">
                              {action.type}
                            </p>
                            <p className="text-sm text-blue-700">
                              Status: {action.result.status}
                            </p>
                          </div>
                        </div>
                      ))}
                    </Tabs.Content>
                  </Tabs.Root>
                ) : (
                  <div className="text-center py-12">
                    <FileText className="mx-auto h-12 w-12 text-gray-400" />
                    <h3 className="mt-2 text-sm font-medium text-gray-900">No results</h3>
                    <p className="mt-1 text-sm text-gray-500">
                      Upload a document to see processing results
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
