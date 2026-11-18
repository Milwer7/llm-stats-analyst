export type ColumnKind = 'numeric' | 'categorical' | 'binary' | 'datetime'

export interface ColumnProfile {
  name: string
  dtype: string
  kind: ColumnKind
  n_missing: number
  n_unique: number
  levels?: string[]
}

export interface DatasetProfile {
  n_rows: number
  n_columns: number
  columns: ColumnProfile[]
}

export interface Dataset {
  id: string
  name: string
  profile: DatasetProfile
  preview: Record<string, unknown>[]
}

export interface AnalysisStep {
  tool: string
  arguments: Record<string, unknown>
  output: Record<string, unknown>
  is_error: boolean
}

export interface AnalysisResult {
  answer: string
  steps: AnalysisStep[]
  provider: string
  model: string
  input_tokens: number
  output_tokens: number
  duration_seconds: number
  completed: boolean
}

export interface Health {
  status: string
  provider: string
  model: string
}
