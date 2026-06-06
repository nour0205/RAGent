import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Database, FileText, Layers3, RefreshCw, Search } from 'lucide-react';
import type { DocumentsResponse, KnowledgeDoc } from '../types';
import { motionProps } from '../lib/constants';
import { getJson } from '../lib/api';
import GlassCard from '../components/ui/GlassCard';
import GradientButton from '../components/ui/GradientButton';
import StatusBanner from '../components/ui/StatusBanner';
import { PremiumInput } from '../components/ui/FormControls';

export default function KnowledgePage({ backendUrl }: { backendUrl: string }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<KnowledgeDoc[]>([]);
  const [query, setQuery] = useState('');

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
  const filteredDocs = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return documents;
    return documents.filter((doc) => `${doc.document_id || ''} ${doc.preview || ''}`.toLowerCase().includes(q));
  }, [documents, query]);

  return (
    <motion.div {...motionProps} className="px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1080px] space-y-6">
        <GlassCard className="p-6 sm:p-8">
          <div className="grid gap-7 lg:grid-cols-[1fr_0.65fr] lg:items-center">
            <div>
              <p className="section-kicker">Library</p>
              <h1 className="mt-3 text-4xl font-semibold tracking-tight text-[#0b0a12]">Your study material.</h1>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-[#6f6878]">
                Notes Recall can use as evidence when answering your questions.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <LibraryStat icon={FileText} value={documents.length} label="notes" />
              <LibraryStat icon={Layers3} value={totalChunks} label="chunks" />
            </div>
          </div>
        </GlassCard>

        <GlassCard className="p-5 sm:p-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="relative max-w-2xl flex-1">
              <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#8a8090]" />
              <PremiumInput value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search your notes..." className="pl-11" />
            </div>
            <GradientButton variant="secondary" onClick={() => void loadDocs()}>
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </GradientButton>
          </div>
          {error ? <div className="mt-5"><StatusBanner tone="warning" title="Library unavailable" message={error} /></div> : null}
        </GlassCard>

        {filteredDocs.length ? (
          <div className="space-y-4">
            {filteredDocs.map((doc, index) => (
              <motion.div key={`${doc.document_id}-${index}`} whileHover={{ y: -2 }}>
                <GlassCard className="p-5 sm:p-6">
                  <div className="flex flex-col gap-5 md:flex-row md:items-start md:justify-between">
                    <div className="min-w-0 flex-1">
                      <div className="mb-3 flex flex-wrap items-center gap-2">
                        <span className="rounded-full bg-[#fbf3f7] px-3 py-1 text-xs font-semibold text-[#9b4f76]">Ready</span>
                        <span className="rounded-full border border-[#eee7ef] bg-white px-3 py-1 text-xs text-[#7b7280]">Study note</span>
                        <span className="rounded-full border border-[#eee7ef] bg-white px-3 py-1 text-xs text-[#7b7280]">{doc.chunks || 0} chunks</span>
                      </div>
                      <h3 className="text-2xl font-semibold tracking-tight text-[#0b0a12]">{prettyTitle(doc.document_id)}</h3>
                      <p className="mt-3 line-clamp-3 max-w-3xl text-sm leading-7 text-[#6f6878]">{doc.preview || 'No preview available.'}</p>
                      <div className="mt-4 flex flex-wrap gap-2">
                        {topicChips(doc).map((topic) => (
                          <span key={topic} className="rounded-full bg-[#fbf7fb] px-3 py-1 text-xs text-[#6f6878]">{topic}</span>
                        ))}
                      </div>
                    </div>
                    <div className="shrink-0 rounded-3xl border border-[#eee7ef] bg-white px-4 py-3 text-sm leading-6 text-[#6f6878]">
                      <p className="font-semibold text-[#0b0a12]">Best use</p>
                      <p>Ask Recall to explain or revise this note.</p>
                    </div>
                  </div>
                  <div className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-[#0b0a12]">
                    Ask from this note <ArrowRight className="h-4 w-4" />
                  </div>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        ) : !loading ? (
          <GlassCard className="p-10 text-center">
            <Database className="mx-auto h-10 w-10 text-[#8a8090]" />
            <h4 className="mt-4 text-xl font-semibold tracking-tight text-[#0b0a12]">No notes found</h4>
            <p className="mx-auto mt-3 max-w-2xl text-sm leading-7 text-[#6f6878]">
              Add notes first, then come back here to browse your study memory.
            </p>
          </GlassCard>
        ) : null}
      </div>
    </motion.div>
  );
}

function LibraryStat({ icon: Icon, value, label }: { icon: any; value: number | string; label: string }) {
  return (
    <div className="rounded-3xl border border-[#eee7ef] bg-white p-4 shadow-sm">
      <Icon className="mb-4 h-5 w-5 text-[#8a8090]" />
      <p className="text-3xl font-semibold tracking-tight text-[#0b0a12]">{value}</p>
      <p className="mt-1 text-xs uppercase tracking-[0.18em] text-[#8a8090]">{label}</p>
    </div>
  );
}

function prettyTitle(value?: string) {
  if (!value) return 'Untitled note';
  return value.replace(/[_-]+/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

function topicChips(doc: KnowledgeDoc) {
  const raw = String(doc.document_id || '').replace(/[_-]+/g, ' ');
  return raw
    .split(/\s+/)
    .filter((part) => part.length > 2)
    .slice(0, 4)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1));
}
