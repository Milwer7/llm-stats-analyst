import type { AnalysisResult, Dataset, Health } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `Request failed with status ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<Health>('/health'),

  uploadDataset: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<Dataset>('/datasets', { method: 'POST', body: form })
  },

  analyze: (datasetId: string, question: string) =>
    request<AnalysisResult>(`/datasets/${datasetId}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    }),
}
