import type { Dataset } from '../api/types'

export function DatasetOverview({ dataset }: { dataset: Dataset }) {
  const { profile, preview } = dataset
  const headers = profile.columns.map((column) => column.name)

  return (
    <section className="card">
      <header className="card-header">
        <h2>{dataset.name}</h2>
        <span className="muted">
          {profile.n_rows.toLocaleString()} rows · {profile.n_columns} columns
        </span>
      </header>

      <div className="chips">
        {profile.columns.map((column) => (
          <span key={column.name} className={`chip kind-${column.kind}`} title={column.dtype}>
            {column.name}
            <small>
              {column.kind}
              {column.n_missing > 0 ? ` · ${column.n_missing} missing` : ''}
            </small>
          </span>
        ))}
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {headers.map((header) => (
                <th key={header}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.map((row, index) => (
              <tr key={index}>
                {headers.map((header) => (
                  <td key={header}>{formatCell(row[header])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return '—'
  return String(value)
}
