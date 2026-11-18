import Markdown from 'react-markdown'
import type { AnalysisResult, AnalysisStep } from '../api/types'

export function AnalysisView({ result }: { result: AnalysisResult }) {
  return (
    <section className="card">
      <header className="card-header">
        <h2>Answer</h2>
        <span className="muted">
          {result.provider}/{result.model} · {result.duration_seconds.toFixed(1)} s ·{' '}
          {(result.input_tokens + result.output_tokens).toLocaleString()} tokens
        </span>
      </header>
      {!result.completed && <p className="warning">The agent stopped at its step limit.</p>}
      <div className="markdown">
        <Markdown>{result.answer || '_No answer was produced._'}</Markdown>
      </div>

      <h3>How it got there</h3>
      <ol className="steps">
        {result.steps.map((step, index) => (
          <StepItem key={index} step={step} />
        ))}
      </ol>
    </section>
  )
}

function StepItem({ step }: { step: AnalysisStep }) {
  const pValue = step.output.p_value
  return (
    <li className={step.is_error ? 'step error' : 'step'}>
      <details>
        <summary>
          <code>{step.tool}</code>
          {typeof pValue === 'number' && <span className="pill">p = {formatP(pValue)}</span>}
          {step.is_error && <span className="pill danger">error, retried</span>}
        </summary>
        <pre>{JSON.stringify({ arguments: step.arguments, output: step.output }, null, 2)}</pre>
      </details>
    </li>
  )
}

function formatP(value: number): string {
  return value < 0.001 ? '< 0.001' : value.toFixed(3)
}
