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
                        {/* Format and Intent */}
                        <div className="flex items-center space-x-2">
                          {getFormatIcon(result.classification.format)}
                          <span className="text-sm font-medium text-gray-900">
                            Format: {result.classification.format}
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
                                  result.classification.intent.confidence >= 0.7 
                                    ? 'bg-green-100 text-green-700'
                                    : result.classification.intent.confidence >= 0.4
                                    ? 'bg-yellow-100 text-yellow-700'
                                    : 'bg-red-100 text-red-700'
                                }`}>
                                  {result.classification.intent.type}
                                </span>
                              </div>
                              <div className="flex items-center justify-between">
                                <span>Confidence:</span>
                                <div className="flex items-center space-x-2">
                                  <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                                    <div 
                                      className={`h-full rounded-full ${
                                        result.classification.intent.confidence >= 0.7 
                                          ? 'bg-green-500'
                                          : result.classification.intent.confidence >= 0.4
                                          ? 'bg-yellow-500'
                                          : 'bg-red-500'
                                      }`}
                                      style={{ width: `${result.classification.intent.confidence * 100}%` }}
                                    />
                                  </div>
                                  <span>{(result.classification.intent.confidence * 100).toFixed(1)}%</span>
                                </div>
                              </div>
                            </div>
                            {result.classification.intent.confidence_explanation && (
                              <div className="mt-2 text-xs text-gray-500">
                                {result.classification.intent.confidence_explanation}
                              </div>
                            )}
                            <div className="mt-3 p-3 bg-gray-50 rounded-md">
                              <p className="text-sm">{result.classification.intent.reasoning}</p>
                            </div>
                          </div>

                          {/* Keywords */}
                          {result.classification.intent.keywords?.length > 0 && (
                            <div className="mt-4">
                              <p className="font-medium text-gray-900">Detected Keywords</p>
                              <div className="flex flex-wrap gap-2 mt-2">
                                {result.classification.intent.keywords.map((keyword: string, index: number) => (
                                  <span 
                                    key={index} 
                                    className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium ${
                                      result.classification.intent.confidence >= 0.7
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

                          {/* Key Entities */}
                          {result.classification.analysis?.key_entities && (
                            <div>
                              <p className="font-medium text-gray-900">Detected Entities</p>
                              <div className="grid grid-cols-2 gap-4 mt-2">
                                {result.classification.analysis.key_entities.monetary_values?.length > 0 && (
                                  <div>
                                    <p className="text-xs font-medium text-gray-500">Monetary Values</p>
                                    <ul className="mt-1 list-disc list-inside">
                                      {result.classification.analysis.key_entities.monetary_values.map((value: string, index: number) => (
                                        <li key={index} className="text-sm">{value}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}
                                {result.classification.analysis.key_entities.dates?.length > 0 && (
                                  <div>
                                    <p className="text-xs font-medium text-gray-500">Dates</p>
                                    <ul className="mt-1 list-disc list-inside">
                                      {result.classification.analysis.key_entities.dates.map((date: string, index: number) => (
                                        <li key={index} className="text-sm">{date}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Routing Information */}
                          <div>
                            <p className="font-medium text-gray-900">Routing Details</p>
                            <div className="mt-2 grid grid-cols-2 gap-4">
                              <div>
                                <p className="text-xs font-medium text-gray-500">Department</p>
                                <p className="mt-1">{result.classification.routing.suggested_department}</p>
                              </div>
                              <div>
                                <p className="text-xs font-medium text-gray-500">Priority</p>
                                <span className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium ${
                                  result.classification.routing.suggested_priority === 'high' 
                                    ? 'bg-red-100 text-red-700'
                                    : 'bg-yellow-100 text-yellow-700'
                                }`}>
                                  {result.classification.routing.suggested_priority.toUpperCase()}
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* Suggested Actions */}
                          {result.classification.analysis?.suggested_actions?.length > 0 && (
                            <div>
                              <p className="font-medium text-gray-900">Suggested Actions</p>
                              <div className="mt-2 space-y-2">
                                {result.classification.analysis.suggested_actions.map((action: any, index: number) => (
                                  <div key={index} className="flex items-start space-x-2 p-2 rounded-md bg-white border border-gray-200">
                                    <div className={`mt-0.5 w-2 h-2 rounded-full ${
                                      action.priority === 'high' ? 'bg-red-500' : 'bg-yellow-500'
                                    }`} />
                                    <div>
                                      <p className="text-sm font-medium">{action.type}</p>
                                      <p className="text-xs text-gray-500">{action.reason}</p>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Document Summary */}
                          {result.classification.analysis?.summary && (
                            <div>
                              <p className="font-medium text-gray-900">Summary</p>
                              <p className="mt-1 text-sm">{result.classification.analysis.summary}</p>
                            </div>
                          )}

                          <div className="mt-4 pt-4 border-t border-gray-200">
                            <p className="text-xs text-gray-500">Memory ID: {result.memory_id}</p>
                          </div>
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
