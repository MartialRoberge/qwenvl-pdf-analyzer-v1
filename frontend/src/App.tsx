import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { DocumentArrowUpIcon, DocumentMagnifyingGlassIcon } from '@heroicons/react/24/outline'
import axios from 'axios'
import clsx from 'clsx'

interface Analysis {
  content: string;
  page_number: number;
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [analyses, setAnalyses] = useState<Analysis[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const onDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0])
      setError(null)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1
  })

  const analyzePDF = async () => {
    if (!file) {
      setError('Please select a PDF file first')
      return
    }

    setIsLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('pdf', file)

    try {
      const response = await axios.post('http://localhost:5004/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      setAnalyses(response.data.analyses)
    } catch (err) {
      setError('An error occurred while analyzing the PDF. Please try again.')
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-finance-50 to-finance-100">
      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <h1 className="text-4xl font-bold text-finance-800 mb-2">
            Financial Document Analyzer
          </h1>
          <p className="text-lg text-finance-600">
            Upload your financial document for expert analysis
          </p>
        </motion.div>

        <motion.div
          className={clsx(
            'mt-12 max-w-3xl mx-auto rounded-xl shadow-lg transition-colors duration-200',
            isDragActive ? 'bg-finance-100 border-finance-400' : 'bg-white border-finance-200',
            'border-2 border-dashed p-8'
          )}
          {...getRootProps()}
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
        >
          <input {...getInputProps()} />
          <div className="text-center">
            <DocumentArrowUpIcon 
              className={clsx(
                'mx-auto h-12 w-12 mb-4',
                isDragActive ? 'text-finance-500' : 'text-finance-400'
              )} 
            />
            {file ? (
              <p className="text-finance-600">
                Selected: <span className="font-semibold">{file.name}</span>
              </p>
            ) : (
              <p className="text-finance-500">
                {isDragActive ? (
                  "Drop your PDF here..."
                ) : (
                  "Drag & drop your financial document, or click to select"
                )}
              </p>
            )}
          </div>
        </motion.div>

        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-4 text-center text-red-600 bg-red-50 py-2 px-4 rounded-md"
          >
            {error}
          </motion.div>
        )}

        <motion.div className="mt-8 text-center">
          <motion.button
            onClick={analyzePDF}
            disabled={!file || isLoading}
            className={clsx(
              'inline-flex items-center px-6 py-3 rounded-lg shadow-md',
              'text-white bg-finance-600 hover:bg-finance-700',
              'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-finance-500',
              'transition-colors duration-200',
              (!file || isLoading) && 'opacity-50 cursor-not-allowed'
            )}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <DocumentMagnifyingGlassIcon className="mr-2 h-5 w-5" />
            {isLoading ? 'Analyzing...' : 'Analyze Document'}
          </motion.button>
        </motion.div>

        <AnimatePresence>
          {analyses.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-12 max-w-4xl mx-auto"
            >
              <h2 className="text-2xl font-bold text-finance-800 mb-6">
                Analysis Results
              </h2>
              {analyses.map((analysis, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="mb-6 bg-white rounded-lg shadow-md overflow-hidden"
                >
                  <div className="bg-finance-50 px-6 py-3 border-b border-finance-100">
                    <h3 className="text-lg font-medium text-finance-700">
                      Page {analysis.page_number}
                    </h3>
                  </div>
                  <div className="px-6 py-4">
                    <p className="text-finance-600 whitespace-pre-wrap">
                      {analysis.content}
                    </p>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {isLoading && (
          <div className="mt-12 max-w-4xl mx-auto">
            <div className="animate-pulse space-y-4">
              {[1, 2, 3].map((n) => (
                <div key={n} className="bg-finance-100 h-32 rounded-lg" />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
