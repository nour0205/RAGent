import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { BookOpen, Database, FileText, Layers3, Link2, RefreshCw } from 'lucide-react';
import type { DocumentsResponse, KnowledgeDoc } from '../types';
import { motionProps } from '../lib/constants';
import { getJson } from '../lib/api';
import { cn } from '../lib/utils';
import GlassCard from '../components/ui/GlassCard';
import StatCard from '../components/ui/StatCard';
import SectionHeading from '../components/ui/SectionHeading';
import GradientButton from '../components/ui/GradientButton';
import StatusBanner from '../components/ui/StatusBanner';
import FeatureChip from '../components/ui/FeatureChip';

export default function KnowledgePage({ backendUrl }: { backendUrl: string }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<KnowledgeDoc[]>([]);

  async function loadDocs() {
    setLoading(true);
    setError(null);
    try {
      const data = await getJson<DocumentsResponse>(`${backendUrl}/documents`);
      setDocuments(data.documents || []);
    } catch (err: any) {
      setError(err?.message || 'Could not load documents.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadDocs();
  }, [backendUrl]);

  const totalChunks = documents.reduce((sum, doc) => sum + Number(doc.chunks || 0), 0);

  return (
    <motion.div {...motionProps} className="px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="grid gap-4 md:grid-cols-3">
          <StatCard title="Ingested Documents" value={String(documents.length)} change="Live corpus count" icon={FileText} tone="from-emerald-500/15 via-cyan-500/10 to-transparent" />
          <StatCard title="Retrieval Ready Chunks" value={String(totalChunks)} change="Retrieval building blocks" icon={Layers3} tone="from-cyan-500/15 via-blue-500/10 to-transparent" />
          <StatCard title="Knowledge State" value={documents.length ? 'Ready' : 'Empty'} change={documents.length ? 'Corpus available' : 'Awaiting ingestion'} icon={Database} tone="from-violet-500/15 via-fuchsia-500/10 to-transparent" />
        </div>

        <GlassCard className="p-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <SectionHeading eyebrow="Corpus Explorer" title={`Knowledge base ${loading ? 'loading' : 'overview'}`} description="Browse the corpus that Recall can retrieve from. Each document card represents source material available for grounded study answers." />
            <GradientButton variant="secondary" onClick={() => void loadDocs()}><RefreshCw className={cn('h-4 w-4', loading && 'animate-spin')} /> Refresh Documents</GradientButton>
          </div>
          {error ? <div className="mt-5"><StatusBanner tone="warning" title="Load failed" message={error} /></div> : null}
        </GlassCard>

        {documents.length ? (
          <div className="grid gap-5 lg:grid-cols-2">
            {documents.map((doc, index) => (
              <motion.div key={`${doc.document_id}-${index}`} whileHover={{ y: -4 }}>
                <GlassCard className="h-full p-6">
                  <div className="mb-5 flex items-start justify-between gap-4">
                    <div>
                      <p className="text-xs uppercase tracking-[0.25em] text-slate-500">Document</p>
                      <h4 className="mt-2 text-xl font-semibold tracking-tight text-slate-900">{doc.document_id || 'unknown'}</h4>
                    </div>
                    <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-700">
                      {doc.chunks || 0} chunk(s)
                    </div>
                  </div>
                  <div className="rounded-[26px] border border-slate-200 bg-slate-50/80 p-4">
                    <p className="mb-2 text-sm font-medium text-slate-700">Preview</p>
                    <p className="text-sm leading-7 text-slate-500">{doc.preview || 'No preview available.'}</p>
                  </div>
                  <div className="mt-5 flex flex-wrap gap-3">
                    <FeatureChip icon={Link2} label="Retrieval Ready" tone="border-sky-200 bg-sky-50 text-sky-700" />
                    <FeatureChip icon={BookOpen} label="Source-backed" tone="border-violet-200 bg-violet-50 text-violet-700" />
                  </div>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        ) : !loading ? (
          <GlassCard className="p-10 text-center">
            <Database className="mx-auto h-10 w-10 text-slate-500" />
            <h4 className="mt-4 text-xl font-semibold tracking-tight text-slate-900">No notes ingested yet</h4>
            <p className="mx-auto mt-3 max-w-2xl text-sm leading-7 text-slate-500">
              Once your backend returns documents from /documents, they will appear here as clean note cards with chunk counts and previews.
            </p>
          </GlassCard>
        ) : null}
      </div>
    </motion.div>
  );
}
