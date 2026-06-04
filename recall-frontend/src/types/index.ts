export type Source = {
  document_id?: string;
  chunk_index?: number | string;
  retrieval_type?: string;
  hybrid_score?: number | string | null;
  text?: string;
};

export type StudyPathItem = {
  topic?: string;
  title?: string;
  priority?: string;
  detail?: string;
};

export type AskResponse = {
  answer?: string;
  route?: string;
  sources?: Source[];
  study_hint?: string;
  concepts?: string[];
  key_concepts?: string[];
  study_path?: StudyPathItem[];
};

export type IngestResponse = {
  status?: string;
  chunks_added?: number;
  [key: string]: unknown;
};

export type KnowledgeDoc = {
  document_id?: string;
  chunks?: number;
  preview?: string;
};

export type DocumentsResponse = {
  documents?: KnowledgeDoc[];
};

export type StatusTone = 'success' | 'warning' | 'info';
export type Page = 'home' | 'ask' | 'ingest' | 'knowledge';
