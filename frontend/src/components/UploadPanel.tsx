import { useRef, useState } from 'react'

interface Props {
  onUpload: (file: File) => Promise<void>
}

export function UploadPanel({ onUpload }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)

  async function handle(file: File | undefined) {
    if (!file) return
    setBusy(true)
    try {
      await onUpload(file)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      className={`dropzone ${dragging ? 'dragging' : ''}`}
      onDragOver={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        void handle(event.dataTransfer.files[0])
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        hidden
        onChange={(event) => void handle(event.target.files?.[0])}
      />
      <strong>{busy ? 'Uploading…' : 'Drop a CSV here or click to choose one'}</strong>
      <span className="muted">
        Try <code>data/samples/checkout_experiment.csv</code> from the repository.
      </span>
    </div>
  )
}
