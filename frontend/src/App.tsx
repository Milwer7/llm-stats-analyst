import { useEffect, useState } from 'react'
import { api } from './api/client'
import type { AnalysisResult, Dataset, Health } from './api/types'
import { AnalysisView } from './components/AnalysisView'
import { DatasetOverview } from './components/DatasetOverview'
import { QuestionForm } from './components/QuestionForm'
import { UploadPanel } from './components/UploadPanel'

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
  }, [])

  async function upload(file: File) {
    setError(null)
    setResult(null)
    try {
      setDataset(await api.uploadDataset(file))
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function ask(question: string) {
    if (!dataset) return
    setError(null)
    setLoading(true)
    try {
      setResult(await api.analyze(dataset.id, question))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="container">
      <header className="app-header">
        <div>
          <h1>LLM Stats Analyst</h1>
          <p className="muted">
            Upload a CSV, ask a question, and an agent picks and runs the right statistical test.
          </p>
        </div>
        <span className={`status ${health ? 'ok' : 'down'}`}>
          {health ? `${health.provider} · ${health.model}` : 'backend offline'}
        </span>
      </header>

      <UploadPanel onUpload={upload} />
      {error && <p className="error-banner">{error}</p>}
      {dataset && <DatasetOverview dataset={dataset} />}
      {dataset && <QuestionForm disabled={loading} onAsk={ask} />}
      {loading && <p className="muted center">The agent is planning and running tests…</p>}
      {result && !loading && <AnalysisView result={result} />}
    </main>
  )
}
