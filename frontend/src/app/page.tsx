'use client'

import { useState, useEffect, useRef } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import * as z from 'zod'
import axios from 'axios'
import * as Progress from '@radix-ui/react-progress'
import * as Separator from '@radix-ui/react-separator'
import * as Tabs from '@radix-ui/react-tabs'
import { Upload, AlertTriangle, FileText, Mail, FileJson, CheckCircle, XCircle, AlertCircle, Loader } from 'lucide-react'

const API_URL = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000'

const formSchema = z.object({
  file: z.instanceof(File)
})

type ProcessingStage = 
  | 'initializing'
  | 'format_detection'
  | 'content_extraction' 
  | 'analysis'
  | 'classification'
  | 'validation'
  | 'completed'
  | 'error'

interface ProcessingProgress {
  stage: ProcessingStage
  progress: number
  details: string
  data?: any
  timestamp: string
}

export default function Home() {
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [processingProgress, setProcessingProgress] = useState<ProcessingProgress | null>(null)
  const [taskId, setTaskId] = useState<string | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })
  
  // Handle WebSocket setup and cleanup
  useEffect(() => {
    if (!taskId) return
    
    // Close any existing connection
    if (socketRef.current) {
      socketRef.current.close()
    }
    
    // Create new WebSocket connection
    const socket = new WebSocket(`${WS_URL}/ws/${taskId}`)
    socketRef.current = socket
    
    socket.onopen = () => {
      console.log('WebSocket connected')
    }
    
    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('WebSocket message:', data)
        
        if (data.type === 'progress') {
          setProcessingProgress(data.data)
        } else if (data.type === 'result') {
          setResult(data.data)
          setProcessingProgress({
            stage: 'completed',
            progress: 1.0,
            details: 'Processing complete',
            timestamp: new Date().toISOString()
          })
          setProcessing(false)
        } else if (data.type === 'error') {
          setError(data.error || 'An error occurred during processing')
          setProcessing(false)
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err)
      }
    }
    
    socket.onerror = (error) => {
      console.error('WebSocket error:', error)
      setError('WebSocket connection error')
    }
    
    socket.onclose = () => {
      console.log('WebSocket closed')
    }
    
    // Cleanup on unmount
    return () => {
      socket.close()
    }
  }, [taskId])
  
  const onSubmit = async (data: z.infer<typeof formSchema>) => {
    try {
      setProcessing(true)
      setError(null)
      setUploadProgress(0)
      setProcessingProgress(null)
      setResult(null)
      setTaskId(null)
      
      const formData = new FormData()
      formData.append('file', data.file)
      
      const response = await axios.post(`${API_URL}/api/upload`, formData, {
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
      
      // Set the task ID to establish WebSocket connection
      if (response.data.task_id) {
        setTaskId(response.data.task_id)
        setProcessingProgress({
          stage: 'initializing',
          progress: 0,
          details: 'Starting processing...',
          timestamp: new Date().toISOString()
        })
      } else {
        throw new Error('No task ID returned from server')
      }
    } catch (err: any) {
      setProcessing(false)
      setError(err.response?.data?.detail || 'An error occurred')
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
  
  const getStageTitle = (stage: ProcessingStage): string => {
    switch (stage) {
      case 'initializing': return 'Initializing'
      case 'format_detection': return 'Detecting Format'
      case 'content_extraction': return 'Extracting Content'
      case 'analysis': return 'Analyzing Content'
      case 'classification': return 'Classifying Document'
      case 'validation': return 'Validating Results'
      case 'completed': return 'Processing Complete'
      case 'error': return 'Error'
      default: return 'Processing'
    }
  }
  
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <h1 className="text-2xl font-bold text-gray-900">AI Document Analysis System</h1>
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
                        Supports PDF, JSON, Email and Text files
                      </span>
                    </label>
                  </div>
                  
                  {form.watch('file') && (
                    <div className="flex items-center space-x-2 text-sm text-gray-600">
                      <FileText className="h-4 w-4" />
                      <span>{form.watch('file').name}</span>
                    </div>
                  )}
                  
                  {/* File Upload Progress */}
                  {processing && uploadProgress < 100 && (
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
                  
                  {/* Processing Progress */}
                  {processing && uploadProgress === 100 && processingProgress && (
                    <div className="space-y-2 mt-4">
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center">
                          <Loader className="h-4 w-4 mr-2 animate-spin text-blue-500" />
                          <span className="font-medium text-gray-700">{getStageTitle(processingProgress.stage)}</span>
                        </div>
                        <span className="text-gray-500">{Math.round(processingProgress.progress * 100)}%</span>
                      </div>
                      <Progress.Root
                        className="h-2 overflow-hidden bg-gray-200 rounded-full"
                        value={processingProgress.progress * 100}
                      >
                        <Progress.Indicator
                          className="h-full bg-blue-500 transition-all duration-300 ease-in-out"
                          style={{ width: `${processingProgress.progress * 100}%` }}
                        />
                      </Progress.Root>
                      <p className="text-xs text-gray-500 mt-1">{processingProgress.details}</p>
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
                        value="analysis"
                        className="px-3 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent data-[state=active]:border-blue-500 data-[state=active]:text-blue-600"
                      >
                        Analysis
                      </Tabs.Trigger>
                      <Tabs.Trigger
                        value="actions"
                        className="px-3 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent data-[state=active]:border-blue-500 data-[state=active]:text-blue-600"
                      >
                        Actions
                      </Tabs.Trigger>
                      <Tabs.Trigger
                        value="details"
                        className="px-3 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent data-[state=active]:border-blue-500 data-[state=active]:text-blue-600"
                      >
                        Details
                      </Tabs.Trigger>
                    </Tabs.List>
                    
                    <Tabs.Content value="classification" className="space-y-4">
                      <div className="rounded-md bg-gray-50 p-4">
                        {/* Format and Intent */}
                        <div className="flex items-center space-x-2">
                          {getFormatIcon(result.format || result.classification?.format || "unknown")}
                          <span className="text-sm font-medium text-gray-900">
                            Format: {result.format || result.classification?.format || "Unknown"}
                          </span>
                        </div>
                        <Separator.Root className="my-4 h-px bg-gray-200" />
                        
                        {/* Intent Details */}
                        <div className="text-sm text-gray-600 space-y-3">
                          <div>
                            <p className="font-medium text-gray-900">Intent Analysis</p>
                            <div className="mt-2 space-y-2">
                              <div className="flex items-center justify-between">
                                <span>Type:</span>
                                <span className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium ${
                                  (result.intent?.confidence || result.classification?.intent?.confidence || 0) >= 0.7 
                                    ? 'bg-green-100 text-green-700'
                                    : (result.intent?.confidence || result.classification?.intent?.confidence || 0) >= 0.4
                                    ? 'bg-yellow-100 text-yellow-700'
                                    : 'bg-red-100 text-red-700'
                                }`}>
                                  {result.intent?.type || result.classification?.intent?.type || "Unknown"}
                                </span>
                              </div>
                              <div className="flex items-center justify-between">
                                <span>Confidence:</span>
                                <div className="flex items-center space-x-2">
                                  <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                                    <div 
                                      className={`h-full rounded-full ${
                                        (result.intent?.confidence || result.classification?.intent?.confidence || 0) >= 0.7 
                                          ? 'bg-green-500'
                                          : (result.intent?.confidence || result.classification?.intent?.confidence || 0) >= 0.4
                                          ? 'bg-yellow-500'
                                          : 'bg-red-500'
                                      }`}
                                      style={{ width: `${(result.intent?.confidence || result.classification?.intent?.confidence || 0) * 100}%` }}
                                    />
                                  </div>
                                  <span>{((result.intent?.confidence || result.classification?.intent?.confidence || 0) * 100).toFixed(1)}%</span>
                                </div>
                              </div>
                            </div>
                            {(result.intent?.confidence_explanation || result.classification?.intent?.confidence_explanation) && (
                              <div className="mt-2 text-xs text-gray-500">
                                {result.intent?.confidence_explanation || result.classification?.intent?.confidence_explanation}
                              </div>
                            )}
                            <div className="mt-3 p-3 bg-gray-50 rounded-md">
                              <p className="text-sm">{result.intent?.reasoning || result.classification?.intent?.reasoning || "No reasoning provided"}</p>
                            </div>
                          </div>

                          {/* Keywords */}
                          {(result.intent?.keywords?.length > 0 || result.classification?.intent?.keywords?.length > 0) && (
                            <div className="mt-4">
                              <p className="font-medium text-gray-900">Detected Keywords</p>
                              <div className="flex flex-wrap gap-2 mt-2">
                                {(result.intent?.keywords || result.classification?.intent?.keywords || []).map((keyword: string, index: number) => (
                                  <span 
                                    key={index} 
                                    className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium ${
                                      (result.intent?.confidence || result.classification?.intent?.confidence || 0) >= 0.7
                                        ? 'bg-green-100 text-green-700'
                                        : 'bg-blue-100 text-blue-700'
                                    }`}
                                  >
                                    {keyword}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Routing Information */}
                          <div>
                            <p className="font-medium text-gray-900">Routing Details</p>
                            <div className="mt-2 grid grid-cols-2 gap-4">
                              <div>
                                <p className="text-xs font-medium text-gray-500">Department</p>
                                <p className="mt-1">{result.routing?.suggested_department || result.classification?.routing?.suggested_department || "General Processing"}</p>
                              </div>
                              <div>
                                <p className="text-xs font-medium text-gray-500">Priority</p>
                                <span className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium ${
                                  (result.routing?.priority || result.classification?.routing?.priority || "normal") === 'high' || 
                                  (result.routing?.priority || result.classification?.routing?.priority || "normal") === 'critical'
                                    ? 'bg-red-100 text-red-700'
                                    : (result.routing?.priority || result.classification?.routing?.priority || "normal") === 'medium'
                                    ? 'bg-yellow-100 text-yellow-700'
                                    : 'bg-blue-100 text-blue-700'
                                }`}>
                                  {((result.routing?.priority || result.classification?.routing?.priority || "normal")).toUpperCase()}
                                </span>
                              </div>
                            </div>
                          </div>

                          <div className="mt-4 pt-4 border-t border-gray-200">
                            <div className="flex justify-between">
                              <p className="text-xs text-gray-500">Task ID: {result.task_id}</p>
                              <p className="text-xs text-gray-500">
                                {new Date(result.timestamp).toLocaleString()}
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </Tabs.Content>
                    
                    <Tabs.Content value="analysis" className="space-y-4">
                      {/* Get analysis data from either root or nested structure */}
                      {((result.analysis || result.classification?.analysis || result.specialized_analysis?.analysis)) && (
                        <div className="space-y-6">
                          {/* Content Analysis */}
                          {(result.analysis?.content_analysis || result.classification?.analysis?.content_analysis || result.specialized_analysis?.content_analysis) && (
                            <div className="bg-gray-50 rounded-md p-4">
                              <h3 className="font-medium text-gray-900 mb-3">Content Analysis</h3>
                              <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                  <p className="text-xs font-medium text-gray-500">Document Length</p>
                                  <p>{(result.analysis?.content_analysis?.length || result.classification?.analysis?.content_analysis?.length || result.specialized_analysis?.content_analysis?.length || 0)} characters</p>
                                </div>
                                <div>
                                  <p className="text-xs font-medium text-gray-500">Readability</p>
                                  <p className="capitalize">{(result.analysis?.content_analysis?.readability?.complexity || result.classification?.analysis?.content_analysis?.readability?.complexity || result.specialized_analysis?.content_analysis?.readability?.complexity || "Unknown")}</p>
                                </div>
                              </div>
                              
                              {/* Entities Section */}
                              {(result.analysis?.content_analysis?.entities || result.classification?.analysis?.content_analysis?.entities || result.specialized_analysis?.content_analysis?.entities) && (
                                <div className="mt-4 pt-4 border-t border-gray-200">
                                  <h4 className="font-medium text-sm text-gray-900 mb-2">Detected Entities</h4>
                                  <div className="grid grid-cols-2 gap-4">
                                    {/* Use the first available entities data */}
                                    {(() => {
                                      const entities = result.analysis?.content_analysis?.entities || 
                                                      result.classification?.analysis?.content_analysis?.entities || 
                                                      result.specialized_analysis?.content_analysis?.entities || {};
                                      
                                      return (
                                        <>
                                          {entities.emails?.length > 0 && (
                                            <div>
                                              <p className="text-xs font-medium text-gray-500">Emails</p>
                                              <ul className="mt-1 list-disc list-inside">
                                                {entities.emails.map((email: string, index: number) => (
                                                  <li key={index} className="text-sm truncate">{email}</li>
                                                ))}
                                              </ul>
                                            </div>
                                          )}
                                          {entities.dates?.length > 0 && (
                                            <div>
                                              <p className="text-xs font-medium text-gray-500">Dates</p>
                                              <ul className="mt-1 list-disc list-inside">
                                                {entities.dates.map((date: string, index: number) => (
                                                  <li key={index} className="text-sm">{date}</li>
                                                ))}
                                              </ul>
                                            </div>
                                          )}
                                          {entities.amounts?.length > 0 && (
                                            <div>
                                              <p className="text-xs font-medium text-gray-500">Amounts</p>
                                              <ul className="mt-1 list-disc list-inside">
                                                {entities.amounts.map((amount: string, index: number) => (
                                                  <li key={index} className="text-sm">{amount}</li>
                                                ))}
                                              </ul>
                                            </div>
                                          )}
                                        </>
                                      );
                                    })()}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                          
                          {/* Recommendations */}
                          {(() => {
                            const recommendations = result.analysis?.recommendations || 
                                                   result.classification?.analysis?.recommendations || 
                                                   result.specialized_analysis?.recommendations || [];
                            
                            return recommendations.length > 0 ? (
                              <div className="bg-gray-50 rounded-md p-4">
                                <h3 className="font-medium text-gray-900 mb-3">Recommendations</h3>
                                <div className="space-y-2">
                                  {recommendations.map((rec: any, index: number) => (
                                    <div key={index} className="flex items-start space-x-2 p-2 rounded-md bg-white border border-gray-200">
                                      <div className={`mt-0.5 w-2 h-2 rounded-full ${
                                        rec.priority === 'high' || rec.priority === 'critical' ? 'bg-red-500' : 
                                        rec.priority === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
                                      }`} />
                                      <div>
                                        <p className="text-sm font-medium">{rec.action}</p>
                                        <p className="text-xs text-gray-500">{rec.reason}</p>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : null;
                          })()}
                        </div>
                      )}
                    </Tabs.Content>

                    <Tabs.Content value="actions" className="space-y-4">
                      <div className="bg-gray-50 rounded-md p-4">
                        <h3 className="font-medium text-gray-900 mb-3">Actions</h3>
                        
                        {result.actions && result.actions.length > 0 ? (
                          <div className="space-y-4">
                            <p className="text-sm text-gray-600">{result.actions.length} actions determined</p>
                            
                            {/* List of actions */}
                            <div className="space-y-3">
                              {result.actions.map((action: any, index: number) => (
                                <div key={index} className="bg-white p-3 rounded-md border border-gray-200">
                                  <div className="flex justify-between items-start">
                                    <div>
                                      <span className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium ${
                                        action.priority === 'critical' ? 'bg-red-100 text-red-700' :
                                        action.priority === 'high' ? 'bg-orange-100 text-orange-700' :
                                        action.priority === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                                        'bg-blue-100 text-blue-700'
                                      }`}>
                                        {action.priority?.toUpperCase() || "NORMAL"}
                                      </span>
                                      <h4 className="mt-2 font-medium text-gray-900">{action.type}</h4>
                                    </div>
                                    
                                    {/* Execution status if available */}
                                    {action.result && (
                                      <span className={`text-xs px-2 py-1 rounded-md ${
                                        action.result.status === 'success' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                                      }`}>
                                        {action.result.status}
                                      </span>
                                    )}
                                  </div>
                                  
                                  <p className="mt-1 text-sm text-gray-600">{action.description || "No description"}</p>
                                  <p className="mt-1 text-xs text-gray-500">{action.reason || "No reason provided"}</p>
                                  
                                  {/* Show execution result if available */}
                                  {action.result && action.result.response && (
                                    <div className="mt-3 pt-3 border-t border-gray-100">
                                      <p className="text-xs font-medium text-gray-500">Response:</p>
                                      <pre className="mt-1 text-xs bg-gray-50 p-2 rounded overflow-auto max-h-20">
                                        {JSON.stringify(action.result.response, null, 2)}
                                      </pre>
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        ) : (
                          <div className="text-center py-6">
                            <p className="text-sm text-gray-500">No actions determined for this document</p>
                          </div>
                        )}
                      </div>
                    </Tabs.Content>
                    
                    <Tabs.Content value="details" className="space-y-4">
                      <div className="space-y-4">
                        {/* Validation Results */}
                        {(result.validation || result.classification?.validation) && (
                          <div className="bg-gray-50 rounded-md p-4">
                            <h3 className="font-medium text-gray-900 mb-3">Validation</h3>
                            <div className="flex items-center mb-3">
                              <div className={`w-3 h-3 rounded-full ${(result.validation?.is_valid || result.classification?.validation?.is_valid) ? 'bg-green-500' : 'bg-red-500'} mr-2`}></div>
                              <span className="text-sm font-medium">
                                {(result.validation?.is_valid || result.classification?.validation?.is_valid) ? 'Valid document' : 'Invalid document - requires review'}
                              </span>
                            </div>
                            
                            {/* Use the first available warnings data */}
                            {(() => {
                              const warnings = result.validation?.warnings || result.classification?.validation?.warnings || [];
                              return warnings.length > 0 ? (
                                <div className="mt-2">
                                  <p className="text-xs font-medium text-gray-500">Warnings</p>
                                  <ul className="mt-1 space-y-1">
                                    {warnings.map((warning: string, index: number) => (
                                      <li key={index} className="flex items-start">
                                        <AlertTriangle className="h-4 w-4 text-yellow-500 mr-1 flex-shrink-0 mt-0.5" />
                                        <span className="text-sm">{warning}</span>
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              ) : null;
                            })()}
                          </div>
                        )}
                        
                        {/* Processing Information */}
                        <div className="bg-gray-50 rounded-md p-4">
                          <h3 className="font-medium text-gray-900 mb-3">Processing Information</h3>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <p className="text-xs font-medium text-gray-500">Original Filename</p>
                              <p className="truncate">{result.original_filename}</p>
                            </div>
                            <div>
                              <p className="text-xs font-medium text-gray-500">Format</p>
                              <p className="capitalize">{result.format || result.classification?.format || "unknown"}</p>
                            </div>
                            {(result.processing_time || result.classification?.processing_time) && (
                              <>
                                <div>
                                  <p className="text-xs font-medium text-gray-500">Processing Time</p>
                                  <p>{(result.processing_time?.duration_ms || result.classification?.processing_time?.duration_ms || 0)}ms</p>
                                </div>
                                <div>
                                  <p className="text-xs font-medium text-gray-500">Cached</p>
                                  <p>{(result.processing_time?.cached || result.classification?.processing_time?.cached) ? 'Yes' : 'No'}</p>
                                </div>
                              </>
                            )}
                          </div>
                          
                          {/* Raw JSON Viewer */}
                          <div className="mt-4 pt-4 border-t border-gray-200">
                            <button 
                              onClick={() => {
                                navigator.clipboard.writeText(JSON.stringify(result, null, 2))
                                  .then(() => alert('JSON copied to clipboard'))
                                  .catch(err => console.error('Error copying: ', err))
                              }}
                              className="text-xs text-blue-600 hover:text-blue-800 underline mb-2"
                            >
                              Copy JSON to clipboard
                            </button>
                            <pre className="bg-gray-100 p-3 rounded-md text-xs text-gray-800 overflow-auto max-h-64">
                              {JSON.stringify(result, null, 2)}
                            </pre>
                          </div>
                        </div>
                      </div>
                    </Tabs.Content>
                  </Tabs.Root>
                ) : (
                  <div className="text-center py-12">
                    {processing && processingProgress ? (
                      <div className="animate-pulse">
                        <Loader className="mx-auto h-12 w-12 text-blue-500 animate-spin" />
                        <h3 className="mt-4 text-sm font-medium text-gray-900">Processing document</h3>
                        <p className="mt-1 text-sm text-gray-500">
                          {processingProgress.details}
                        </p>
                      </div>
                    ) : (
                      <>
                        <FileText className="mx-auto h-12 w-12 text-gray-400" />
                        <h3 className="mt-2 text-sm font-medium text-gray-900">No results</h3>
                        <p className="mt-1 text-sm text-gray-500">
                          Upload a document to see processing results
                        </p>
                      </>
                    )}
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
