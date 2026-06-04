import { useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import type { IngestResponse } from '../types';
import { motionProps } from '../lib/constants';
import { postJson } from '../lib/api';
import { cn } from '../lib/utils';
import GlassCard from '../components/ui/GlassCard';
import SectionHeading from '../components/ui/SectionHeading';
import GradientButton from '../components/ui/GradientButton';
import StatusBanner from '../components/ui/StatusBanner';
import { Field, PremiumInput, PremiumTextarea } from '../components/ui/FormControls';

export default function IngestPage({ backendUrl }: { backendUrl: string }) {
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({
    document_id: '',
    document_type: '',
    course: '',
    source: '',
    owner: '',
    topic_tags_raw: '',
    text: '',
  });

  const topicTags = useMemo(() => form.topic_tags_raw.split(',').map((tag) => tag.trim()).filter(Boolean), [form.topic_tags_raw]);
  const progress = step === 1 ? 33 : step === 2 ? 67 : 100;

  function update<K extends keyof typeof form>(key: K, value: (typeof form)[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function handleIngest() {
    if (!form.document_id.trim()) {
      setError('Please enter a document ID.');
      return;
    }
    if (!form.text.trim()) {
      setError('Please enter document text.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const payload: Record<string, unknown> = {
        document_id: form.document_id.trim(),
        text: form.text.trim(),
      };
      if (form.document_type.trim()) payload.document_type = form.document_type.trim();
      if (form.course.trim()) payload.course = form.course.trim();
      if (form.source.trim()) payload.source = form.source.trim();
      if (form.owner.trim()) payload.owner = form.owner.trim();
      if (topicTags.length) payload.topic_tags = topicTags;

      const data = await postJson<IngestResponse>(`${backendUrl}/ingest`, payload);
      setResult(data);
      setStep(3);
    } catch (err: any) {
      setError(err?.message || 'Ingestion failed.');
    } finally {
      setLoading(false);
    }
  }

  const steps = [
    { id: 1, title: 'Document Identity', desc: 'Core metadata and study ownership' },
    { id: 2, title: 'Content & Tags', desc: 'Paste notes and structure topic signal' },
    { id: 3, title: 'Review Result', desc: 'Inspect backend ingestion response' },
  ];

  return (
    <motion.div {...motionProps} className="px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
          <GlassCard className="p-6">
            <SectionHeading
              eyebrow="Multi-step Wizard"
              title="Ingest study material"
              description="This flow preserves your existing metadata schema: document_id, document_type, course, source, owner, topic_tags, and document text."
            />
            <div className="mt-6 space-y-4">
              {steps.map((item) => (
                <div key={item.id} className={cn('rounded-[26px] border p-4 transition', step === item.id ? 'border-slate-200 bg-white/85' : 'border-slate-200 bg-white/70')}>
                  <div className="flex items-start gap-4">
                    <div className={cn('flex h-10 w-10 items-center justify-center rounded-2xl text-sm font-semibold', step >= item.id ? 'bg-gradient-to-br from-cyan-400 to-violet-500 text-slate-950' : 'bg-white/85 text-slate-500')}>
                      {item.id}
                    </div>
                    <div>
                      <p className="font-medium text-slate-900">{item.title}</p>
                      <p className="mt-1 text-sm text-slate-500">{item.desc}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-6">
              <div className="mb-3 flex items-center justify-between text-sm">
                <span className="text-slate-500">Workflow progress</span>
                <span className="text-slate-900">{progress}%</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-white/85">
                <motion.div initial={{ width: 0 }} animate={{ width: `${progress}%` }} className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-violet-500 to-fuchsia-400" />
              </div>
            </div>
          </GlassCard>

          <GlassCard className="p-6 sm:p-7">
            <AnimatePresence mode="wait">
              {step === 1 && (
                <motion.div key="step1" initial={{ opacity: 0, x: 18 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -18 }} className="space-y-5">
                  <SectionHeading eyebrow="Step 1" title="Document identity" description="Start by defining the study asset and its surrounding metadata context." />
                  <div className="grid gap-4 md:grid-cols-2">
                    <Field label="Document ID"><PremiumInput value={form.document_id} onChange={(e) => update('document_id', e.target.value)} placeholder="ml_supervised_learning_basics" /></Field>
                    <Field label="Document Type"><PremiumInput value={form.document_type} onChange={(e) => update('document_type', e.target.value)} placeholder="lecture_note" /></Field>
                    <Field label="Course"><PremiumInput value={form.course} onChange={(e) => update('course', e.target.value)} placeholder="Introduction to Machine Learning" /></Field>
                    <Field label="Source"><PremiumInput value={form.source} onChange={(e) => update('source', e.target.value)} placeholder="lecture slides" /></Field>
                  </div>
                  <Field label="Owner"><PremiumInput value={form.owner} onChange={(e) => update('owner', e.target.value)} placeholder="nour" /></Field>
                  <div className="flex gap-3">
                    <GradientButton onClick={() => setStep(2)}>Continue <ArrowRight className="h-4 w-4" /></GradientButton>
                  </div>
                </motion.div>
              )}

              {step === 2 && (
                <motion.div key="step2" initial={{ opacity: 0, x: 18 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -18 }} className="space-y-5">
                  <SectionHeading eyebrow="Step 2" title="Content and topic signals" description="Paste the note content and enrich it with topic tags to improve retrieval quality." />
                  <Field label="Topic Tags" hint="Comma separated">
                    <PremiumInput value={form.topic_tags_raw} onChange={(e) => update('topic_tags_raw', e.target.value)} placeholder="machine learning, supervised learning, generalization" />
                  </Field>
                  <div className="flex flex-wrap gap-2">
                    {topicTags.length ? topicTags.map((tag) => (
                      <span key={tag} className="rounded-full border border-violet-200 bg-violet-50 px-3 py-1.5 text-xs text-violet-700">{tag}</span>
                    )) : <span className="text-sm text-slate-500">Topic chips appear here.</span>}
                  </div>
                  <Field label="Document Text">
                    <PremiumTextarea rows={12} value={form.text} onChange={(e) => update('text', e.target.value)} placeholder="Paste the study material here..." />
                  </Field>
                  {error ? <StatusBanner tone="warning" title="Validation issue" message={error} /> : null}
                  <div className="flex flex-wrap gap-3">
                    <GradientButton variant="secondary" onClick={() => setStep(1)}>Back</GradientButton>
                    <GradientButton onClick={handleIngest} disabled={loading}>{loading ? 'Ingesting study material...' : 'Ingest'}</GradientButton>
                  </div>
                </motion.div>
              )}

              {step === 3 && (
                <motion.div key="step3" initial={{ opacity: 0, x: 18 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -18 }} className="space-y-5">
                  <SectionHeading eyebrow="Step 3" title="Ingestion review" description="A refined result display for backend statuses including ingested, duplicate, conflict, and no content." />
                  {result?.status === 'ingested' ? (
                    <StatusBanner tone="success" title="Document ingested successfully" message={`Chunks added: ${result?.chunks_added ?? 0}`} />
                  ) : result?.status === 'duplicate' ? (
                    <StatusBanner tone="warning" title="Duplicate content" message="This document content was already ingested." />
                  ) : result?.status === 'conflict' ? (
                    <StatusBanner tone="warning" title="Document ID conflict" message="That document_id already exists." />
                  ) : result?.status === 'no content' ? (
                    <StatusBanner tone="warning" title="No content extracted" message="No content could be extracted from the document." />
                  ) : result ? (
                    <StatusBanner tone="info" title="Backend response received" message="The backend returned a custom ingestion payload." />
                  ) : (
                    <StatusBanner tone="info" title="No result yet" message="Submit the ingestion form to see a structured response." />
                  )}

                  <GlassCard className="border-slate-200/60 bg-slate-50/80 p-5">
                    <pre className="overflow-x-auto whitespace-pre-wrap text-sm leading-7 text-slate-600">{JSON.stringify(result, null, 2)}</pre>
                  </GlassCard>

                  <div className="flex flex-wrap gap-3">
                    <GradientButton onClick={() => { setResult(null); setError(null); setStep(1); }}>Ingest another document</GradientButton>
                    <GradientButton variant="secondary" onClick={() => setStep(2)}>Edit inputs</GradientButton>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </GlassCard>
        </div>
      </div>
    </motion.div>
  );
}
