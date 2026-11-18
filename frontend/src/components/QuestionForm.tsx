import { useState } from 'react'

const EXAMPLES = [
  'Did the new checkout increase conversion compared to control?',
  'Is the order value different between variants? The data looks skewed.',
  'Does conversion depend on the device?',
  'Is session length correlated with order value?',
]

interface Props {
  disabled: boolean
  onAsk: (question: string) => void
}

export function QuestionForm({ disabled, onAsk }: Props) {
  const [question, setQuestion] = useState('')

  return (
    <section className="card">
      <form
        className="question-form"
        onSubmit={(event) => {
          event.preventDefault()
          if (question.trim().length >= 3) onAsk(question.trim())
        }}
      >
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask a question about your data…"
          rows={3}
        />
        <button type="submit" disabled={disabled || question.trim().length < 3}>
          {disabled ? 'Analyzing…' : 'Analyze'}
        </button>
      </form>
      <div className="examples">
        {EXAMPLES.map((example) => (
          <button key={example} type="button" className="ghost" onClick={() => setQuestion(example)}>
            {example}
          </button>
        ))}
      </div>
    </section>
  )
}
