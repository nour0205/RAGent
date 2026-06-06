import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, CheckCircle2 } from 'lucide-react';
import type { IngestResponse } from '../types';
import { motionProps } from '../lib/constants';
import { postJson } from '../lib/api';
import GlassCard from '../components/ui/GlassCard';
import GradientButton from '../components/ui/GradientButton';
import StatusBanner from '../components/ui/StatusBanner';
import { Field, PremiumInput, PremiumTextarea } from '../components/ui/FormControls';

export default function IngestPage({ backendUrl }: { backendUrl: string }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({
    document_id: '',
    course: '',
    topic_tags_raw: '',
    text: '',
  });

  const topicTags = useMemo(
    () => form.topic_tags_raw.split(',').map((tag) => tag.trim()).filter(Boolean),
    [form.topic_tags_raw],
  );

  const wordCount = useMemo(() => form.text.trim().split(/\s+/).filter(Boolean).length, [form.text]);

  function update<K extends keyof typeof form>(key: K, value: (typeof form)[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function resetForm() {
    setResult(null);
    setError(null);
    setForm({
      document_id: '',
      course: '',
      topic_tags_raw: '',
      text: '',
    });
  }

  async function handleIngest() {
    if (!form.document_id.trim()) {
      setError('Give this note a short name, for example db_normalization_intro.');
      return;
    }
    if (!form.text.trim()) {
      setError('Paste the note content before adding it to Recall.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const payload: Record<string, unknown> = {
        document_id: form.document_id.trim(),
        document_type: 'lecture_note',
        text: form.text.trim(),
      };
      if (form.course.trim()) payload.course = form.course.trim();
      if (topicTags.length) payload.topic_tags = topicTags;

      const data = await postJson<IngestResponse>(`${backendUrl}/ingest`, payload);
      setResult(data);
    } catch (err: any) {
      setError(err?.message || 'Ingestion failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <motion.div {...motionProps} className="px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[960px] space-y-6">
        <GlassCard className="p-6 sm:p-8">
          <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="section-kicker">Add notes</p>
              <h1 className="mt-3 text-4xl font-semibold tracking-tight text-[#0b0a12]">Add study material.</h1>
              <p className="mt-3 max-w-xl text-sm leading-7 text-[#6f6878]">
                Paste a lecture, summary, or textbook section. Recall will make it searchable.
              </p>
            </div>
            <span className="w-fit rounded-full border border-[#eee7ef] bg-white px-3 py-2 text-xs text-[#8a8090]">{wordCount} words</span>
          </div>

          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <Field label="Title">
                <PremiumInput value={form.document_id} onChange={(e) => update('document_id', e.target.value)} placeholder="db_normalization_intro" />
              </Field>
              <Field label="Course">
                <PremiumInput value={form.course} onChange={(e) => update('course', e.target.value)} placeholder="Database Systems" />
              </Field>
            </div>

            <Field label="Topics">
              <PremiumInput value={form.topic_tags_raw} onChange={(e) => update('topic_tags_raw', e.target.value)} placeholder="normalization, 1NF, anomalies" />
            </Field>

            <Field label="Content">
              <PremiumTextarea rows={15} value={form.text} onChange={(e) => update('text', e.target.value)} placeholder="Paste your notes here..." />
            </Field>

            {error ? <StatusBanner tone="warning" title="Before adding this note" message={error} /> : null}
            {result ? <IngestResult result={result} /> : null}

            <div className="flex flex-wrap gap-3 pt-1">
              <GradientButton onClick={handleIngest} disabled={loading}>
                {loading ? 'Adding to Recall...' : 'Add to library'}
                {!loading && <ArrowRight className="h-4 w-4" />}
              </GradientButton>
              <GradientButton variant="secondary" onClick={resetForm}>Clear</GradientButton>
            </div>
          </div>
        </GlassCard>
      </div>
    </motion.div>
  );
}

function IngestResult({ result }: { result: IngestResponse }) {
  if (result.status === 'ingested') {
    return (
      <div className="rounded-3xl border border-[#ead6e2] bg-[#fbf7fb] p-4 text-[#6f3d7b]">
        <div className="flex items-center gap-2 font-semibold"><CheckCircle2 className="h-4 w-4" /> Added to your library</div>
        <p className="mt-2 text-sm leading-6">Recall created {result.chunks_added ?? 0} searchable chunk(s).</p>
      </div>
    );
  }

  const messages: Record<string, string> = {
    duplicate: 'This exact content is already in your study memory.',
    conflict: 'A note with this title already exists. Choose another title.',
    'no content': 'Recall could not create chunks from this text.',
  };

  return <StatusBanner tone="warning" title="Note not added" message={messages[result.status || ''] || 'The backend returned a custom ingestion status.'} />;
}
