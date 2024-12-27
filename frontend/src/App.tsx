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
    <div className="min-h-screen bg-gradient-to-br from-finance-50 via-white to-finance-100">
      <div className="max-w-7xl mx-auto py-16 px-4 sm:px-6 lg:px-8">
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-4"
        >
          <h1 className="text-5xl font-extrabold text-finance-800 mb-4 bg-clip-text text-transparent bg-gradient-to-r from-finance-600 to-finance-800">
            PDF Analyzer
          </h1>
          <p className="text-xl text-finance-600 max-w-2xl mx-auto">
            Upload your PDF document and let our AI analyze it for you
          </p>
        </motion.div>

        <motion.div
          className={clsx(
            'mt-12 max-w-3xl mx-auto rounded-2xl shadow-xl transition-all duration-300',
            isDragActive ? 'bg-finance-50 border-finance-400 scale-105' : 'bg-white border-finance-200 hover:shadow-2xl hover:border-finance-300',
            'border-2 border-dashed p-12'
          )}
          {...getRootProps()}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <input {...getInputProps()} />
          <div className="text-center space-y-4">
            <DocumentArrowUpIcon 
              className={clsx(
                'mx-auto h-16 w-16 mb-4 transition-colors duration-300',
                isDragActive ? 'text-finance-500' : 'text-finance-400'
              )} 
            />
            {file ? (
              <div className="space-y-2">
                <p className="text-lg text-finance-700 font-medium">
                  Selected file:
                </p>
                <p className="text-finance-600 bg-finance-50 py-2 px-4 rounded-lg inline-block">
                  {file.name}
                </p>
              </div>
            ) : (
              <p className="text-lg text-finance-600">
                {isDragActive ? (
                  "Drop your PDF here to analyze..."
                ) : (
                  "Drag & drop your PDF, or click to select"
                )}
              </p>
            )}
          </div>
        </motion.div>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 text-center text-red-600 bg-red-50 py-3 px-6 rounded-xl max-w-2xl mx-auto border border-red-100"
          >
            {error}
          </motion.div>
        )}

        <motion.div className="mt-8 text-center">
          <motion.button
            onClick={analyzePDF}
            disabled={!file || isLoading}
            className={clsx(
              'inline-flex items-center px-8 py-4 rounded-xl shadow-lg',
              'text-white bg-gradient-to-r from-finance-600 to-finance-700',
              'hover:from-finance-700 hover:to-finance-800',
              'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-finance-500',
              'transition-all duration-300 text-lg font-medium',
              (!file || isLoading) && 'opacity-50 cursor-not-allowed'
            )}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <DocumentMagnifyingGlassIcon className="mr-3 h-6 w-6" />
            {isLoading ? 'Analyzing...' : 'Analyze Document'}
          </motion.button>
        </motion.div>

        <AnimatePresence>
          {analyses.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-16 max-w-4xl mx-auto space-y-8"
            >
              <h2 className="text-3xl font-bold text-finance-800 text-center mb-8">
                Analysis Results
              </h2>
              {analyses.map((analysis, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white rounded-xl shadow-lg overflow-hidden transform transition-all duration-300 hover:shadow-xl"
                >
                  <div className="bg-gradient-to-r from-finance-50 to-white px-8 py-4 border-b border-finance-100">
                    <h3 className="text-xl font-semibold text-finance-700">
                      Page {analysis.page_number}
                    </h3>
                  </div>
                  <div className="px-8 py-6">
                    <p className="text-finance-600 whitespace-pre-wrap text-lg leading-relaxed">
                      {analysis.content}
                    </p>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {isLoading && (
          <div className="mt-16 max-w-4xl mx-auto space-y-6">
            <div className="animate-pulse space-y-6">
              {[1, 2, 3].map((n) => (
                <div key={n} className="bg-finance-100 h-40 rounded-xl" />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
