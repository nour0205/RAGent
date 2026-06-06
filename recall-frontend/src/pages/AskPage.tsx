import { useState, type ReactNode } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ArrowRight, Quote, Sparkles } from 'lucide-react';
import type { AskResponse, Source } from '../types';
import { motionProps } from '../lib/constants';
import { postJson } from '../lib/api';
import { sourceReason, sourceTitle } from '../lib/studyInsights';
import GlassCard from '../components/ui/GlassCard';
import GradientButton from '../components/ui/GradientButton';
import StatusBanner from '../components/ui/StatusBanner';
import { Field, PremiumInput, PremiumTextarea } from '../components/ui/FormControls';

export default function AskPage({ backendUrl }: { backendUrl: string }) {
  const [question, setQuestion] = useState('');
  const [owner, setOwner] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [activeSourceIndex, setActiveSourceIndex] = useState<number | null>(null);
  const [generatedMs, setGeneratedMs] = useState<number | null>(null);

  async function handleAsk() {
    if (!question.trim()) {
      setError('Please enter a question.');
      return;
    }

    setLoading(true);
    setError(null);
    setGeneratedMs(null);

    try {
      const startedAt = performance.now();
      const payload: Record<string, unknown> = { question: question.trim() };
      if (owner.trim()) payload.owner = owner.trim();

      const data = await postJson<AskResponse>(`${backendUrl}/ask`, payload);
      setGeneratedMs(Math.round(performance.now() - startedAt));
      setResult(data);
    } catch (err: any) {
      setError(err?.message || 'Request failed.');
    } finally {
      setLoading(false);
    }
  }

  const sourceCount = result?.sources?.length || 0;
  const documentCount = new Set((result?.sources || []).map((source) => source.document_id).filter(Boolean)).size;

  return (
    <motion.div {...motionProps} className="px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1080px] space-y-6">
        {!result ? (
          <GlassCard className="overflow-hidden px-6 py-8 sm:px-8 lg:px-10">
            <div className="grid gap-8 lg:grid-cols-[0.8fr_1fr] lg:items-center">
              <div>
                <p className="section-kicker">Study brief</p>
                <h1 className="mt-4 max-w-xl text-4xl font-semibold tracking-[-0.055em] text-[#0b0a12] sm:text-5xl">
                  Ask your notes. Get a grounded answer.
                </h1>
                <p className="mt-5 max-w-lg text-base leading-8 text-[#6f6878]">
                  Recall turns your study material into a clear answer with clickable evidence.
                </p>
              </div>

              <QuestionForm
                question={question}
                owner={owner}
                loading={loading}
                error={error}
                setQuestion={setQuestion}
                setOwner={setOwner}
                handleAsk={handleAsk}
              />
            </div>
          </GlassCard>
        ) : (
          <GlassCard className="p-5 sm:p-6">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex items-start gap-4">
                <div className="mt-1 rounded-2xl bg-[#fbf3f7] p-2 text-[#b85f8b]">
                  <Quote className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-sm font-medium text-[#8a8090]">Question</p>
                  <h1 className="mt-1 text-2xl font-semibold tracking-tight text-[#0b0a12]">
                    {question}
                  </h1>
                </div>
              </div>

              <button
                onClick={() => setResult(null)}
                className="w-fit rounded-full border border-[#eee7ef] bg-white px-4 py-2 text-sm font-medium text-[#514858] transition hover:bg-[#fbf7fb]"
              >
                Ask another
              </button>
            </div>
          </GlassCard>
        )}

        <AnimatePresence mode="wait">
          <motion.div
            key={result ? 'result' : 'empty'}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.28 }}
          >
            {result ? (
              <div className="space-y-6">
                <GlassCard className="p-5 sm:p-8">
                  <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <div className="mb-3 inline-flex items-center gap-2 rounded-full bg-[#fbf3f7] px-3 py-1.5 text-xs font-semibold text-[#9b4f76]">
                        <Sparkles className="h-3.5 w-3.5" />
                        Grounded in {sourceCount} source{sourceCount === 1 ? '' : 's'}
                      </div>
                      <h2 className="text-3xl font-semibold tracking-tight text-[#0b0a12]">
                        Study Brief
                      </h2>
                    </div>
                    <div className="flex flex-wrap gap-2 text-xs text-[#7b7280]">
                      <span className="rounded-full border border-[#eee7ef] bg-white px-3 py-1.5">
                        {documentCount || sourceCount} note{(documentCount || sourceCount) === 1 ? '' : 's'} used
                      </span>
                      <span className="rounded-full border border-[#eee7ef] bg-white px-3 py-1.5">
                        {sourceCount} chunk{sourceCount === 1 ? '' : 's'} read
                      </span>
                      {generatedMs ? (
                        <span className="rounded-full border border-[#eee7ef] bg-white px-3 py-1.5">
                          {(generatedMs / 1000).toFixed(1)}s
                        </span>
                      ) : null}
                    </div>
                  </div>

                  <FormattedAnswer
                    text={result.answer || 'No answer returned.'}
                    sources={result.sources || []}
                    onCitationClick={setActiveSourceIndex}
                  />
                </GlassCard>

                <GlassCard className="p-5 sm:p-7">
                  <div className="mb-5 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
                    <div>
                      <h2 className="text-2xl font-semibold tracking-tight text-[#0b0a12]">
                        Sources
                      </h2>
                      <p className="mt-1 text-sm text-[#6f6878]">
                        Open any source to verify the passages behind the brief.
                      </p>
                    </div>
                  </div>

                  <div className="grid gap-3 md:grid-cols-2">
                    {result.sources?.length ? (
                      result.sources.map((source, index) => (
                        <SourceCard
                          key={`${source.document_id}-${index}`}
                          source={source}
                          index={index}
                          onOpen={() => setActiveSourceIndex(index)}
                        />
                      ))
                    ) : (
                      <p className="rounded-3xl border border-[#eee7ef] bg-white p-4 text-sm text-[#6f6878]">
                        No sources returned.
                      </p>
                    )}
                  </div>
                </GlassCard>
              </div>
            ) : null}
          </motion.div>
        </AnimatePresence>
      </div>

      {result && activeSourceIndex !== null ? (
        <SourceModal
          source={result.sources?.[activeSourceIndex]}
          index={activeSourceIndex}
          onClose={() => setActiveSourceIndex(null)}
        />
      ) : null}
    </motion.div>
  );
}

function QuestionForm({
  question,
  owner,
  loading,
  error,
  setQuestion,
  setOwner,
  handleAsk,
}: {
  question: string;
  owner: string;
  loading: boolean;
  error: string | null;
  setQuestion: (value: string) => void;
  setOwner: (value: string) => void;
  handleAsk: () => void;
}) {
  return (
    <div className="rounded-[32px] border border-[#eee7ef] bg-white p-5 shadow-[0_20px_60px_rgba(55,38,62,0.045)] sm:p-6">
      <div className="mb-5">
        <h2 className="text-2xl font-semibold tracking-tight text-[#0b0a12]">
          What do you want to understand?
        </h2>
      </div>

      <div className="space-y-4">
        <Field label="Question">
          <PremiumTextarea
            rows={6}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Example: What should I revise before my machine learning exam?"
          />
        </Field>

        <details className="rounded-2xl border border-[#eee7ef] bg-white px-4 py-3">
          <summary className="cursor-pointer list-none text-sm font-medium text-[#514858]">
            Optional filter
          </summary>
          <div className="mt-3 max-w-sm">
            <Field label="Owner">
              <PremiumInput
                value={owner}
                onChange={(e) => setOwner(e.target.value)}
                placeholder="nour"
              />
            </Field>
          </div>
        </details>

        {error ? <StatusBanner tone="warning" title="Request issue" message={error} /> : null}

        <div className="flex flex-wrap gap-3 pt-1">
          <GradientButton onClick={handleAsk} disabled={loading}>
            {loading ? 'Reading your notes...' : 'Create Study Brief'}
            {!loading && <ArrowRight className="h-4 w-4" />}
          </GradientButton>
          <GradientButton
            variant="secondary"
            onClick={() => setQuestion('What should I revise before my machine learning exam?')}
          >
            Demo question
          </GradientButton>
        </div>
      </div>
    </div>
  );
}

function FormattedAnswer({
  text,
  sources,
  onCitationClick,
}: {
  text: string;
  sources: Source[];
  onCitationClick: (index: number) => void;
}) {
  const normalized = text.replace(/\r/g, '');

  const citationLabel = (sourceIndex: number) => {
    const title = sourceTitle(sources[sourceIndex] || {}, sourceIndex);
    return title.length > 24 ? `${title.slice(0, 24)}…` : title;
  };

  const renderInline = (value: string) => {
    const parts = value.split(/(\*\*[^*]+\*\*|\[S\d+\])/g);

    return parts.map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={index} className="font-semibold text-[#0b0a12]">
            {part.slice(2, -2)}
          </strong>
        );
      }

      const citationMatch = part.match(/^\[S(\d+)\]$/);
      if (citationMatch) {
        const sourceIndex = Number(citationMatch[1]) - 1;
        return (
          <button
            key={index}
            type="button"
            onClick={() => onCitationClick(sourceIndex)}
            className="mx-1 inline-flex translate-y-[-1px] items-center rounded-full bg-[#fbf3f7] px-2.5 py-1 text-[11px] font-semibold leading-none text-[#9b4f76] ring-1 ring-[#ead6e2] transition hover:bg-[#f4e3ed] hover:text-[#7f3f61]"
            aria-label={`Open ${citationLabel(sourceIndex)}`}
          >
            {citationLabel(sourceIndex)}
          </button>
        );
      }

      return <span key={index}>{part}</span>;
    });
  };

  const lines = normalized.split('\n').map((line) => line.trim()).filter(Boolean);
  const elements: ReactNode[] = [];
  let listItems: string[] = [];

  const flushList = () => {
    if (!listItems.length) return;
    elements.push(
      <ul key={`list-${elements.length}`} className="my-5 space-y-3">
        {listItems.map((item, index) => (
          <li key={index} className="flex gap-3 text-[16px] leading-8 text-[#433a49]">
            <span className="mt-[13px] h-1.5 w-1.5 shrink-0 rounded-full bg-[#b85f8b]" />
            <span>{renderInline(item)}</span>
          </li>
        ))}
      </ul>,
    );
    listItems = [];
  };

  lines.forEach((line) => {
    if (line.startsWith('## ')) {
      flushList();
      elements.push(
        <h2 key={`h2-${elements.length}`} className="mt-8 text-2xl font-semibold tracking-tight text-[#0b0a12] first:mt-0">
          {line.replace(/^##\s+/, '')}
        </h2>,
      );
      return;
    }

    if (line.startsWith('### ')) {
      flushList();
      elements.push(
        <h3 key={`h3-${elements.length}`} className="mt-6 text-lg font-semibold tracking-tight text-[#0b0a12]">
          {line.replace(/^###\s+/, '')}
        </h3>,
      );
      return;
    }

    if (/^(-|•|\d+[.)])\s+/.test(line)) {
      listItems.push(line.replace(/^(-|•|\d+[.)])\s+/, ''));
      return;
    }

    flushList();
    elements.push(
      <p key={`p-${elements.length}`} className="my-4 text-[16px] leading-8 text-[#433a49]">
        {renderInline(line)}
      </p>,
    );
  });

  flushList();

  return (
    <article className="rounded-[28px] border border-[#eee7ef] bg-white px-6 py-6 sm:px-8 sm:py-7">
      <div className="max-w-none">{elements}</div>
    </article>
  );
}

function SourceCard({ source, index, onOpen }: { source: Source; index: number; onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="group rounded-3xl border border-[#eee7ef] bg-white p-4 text-left transition hover:-translate-y-0.5 hover:border-[#ead6e2] hover:shadow-[0_18px_45px_rgba(55,38,62,0.05)]"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-[#0b0a12]">{sourceTitle(source, index)}</p>
          <p className="mt-2 line-clamp-2 text-sm leading-6 text-[#6f6878]">{sourceReason(source)}</p>
        </div>
        <span className="shrink-0 rounded-full bg-[#fbf3f7] px-2.5 py-1 text-xs font-semibold text-[#9b4f76]">
          Source
        </span>
      </div>
    </button>
  );
}

function SourceModal({
  source,
  index,
  onClose,
}: {
  source: Source | undefined;
  index: number;
  onClose: () => void;
}) {
  if (!source) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4 backdrop-blur-sm">
      <div className="w-full max-w-2xl rounded-[32px] border border-[#eee7ef] bg-white p-6 shadow-[0_30px_90px_rgba(20,10,30,0.22)]">
        <div className="mb-5 flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-[#9b6680]">Source</p>
            <h3 className="mt-1 text-2xl font-semibold tracking-tight text-[#0b0a12]">
              {sourceTitle(source, index)}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="rounded-full border border-[#eee7ef] px-3 py-1.5 text-sm font-medium text-[#514858] transition hover:bg-[#fbf7fb]"
          >
            Close
          </button>
        </div>
        <div className="max-h-[62vh] overflow-auto rounded-3xl bg-[#fbf7fb] p-5 text-[15px] leading-8 text-[#433a49]">
          {source.text || 'No source text available.'}
        </div>
      </div>
    </div>
  );
}
