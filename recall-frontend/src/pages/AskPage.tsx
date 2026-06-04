import { useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  GraduationCap,
  Layers3,
  Map,
  Route,
  SearchCheck,
  ShieldCheck,
  Sparkles,
  Target,
} from 'lucide-react';
import type { AskResponse } from '../types';
import { motionProps } from '../lib/constants';
import { routeLabel } from '../lib/utils';
import { postJson } from '../lib/api';
import {
  buildStudyPath,
  confidenceLabel,
  extractConcepts,
  routeDescription,
  sourceReason,
  sourceTitle,
} from '../lib/studyInsights';
import GlassCard from '../components/ui/GlassCard';
import StatCard from '../components/ui/StatCard';
import SectionHeading from '../components/ui/SectionHeading';
import FeatureChip from '../components/ui/FeatureChip';
import GradientButton from '../components/ui/GradientButton';
import StatusBanner from '../components/ui/StatusBanner';
import { Field, PremiumInput, PremiumTextarea } from '../components/ui/FormControls';

export default function AskPage({ backendUrl }: { backendUrl: string }) {
  const [question, setQuestion] = useState('');
  const [owner, setOwner] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);

  const concepts = useMemo(() => {
    if (result?.key_concepts?.length) return result.key_concepts;
    if (result?.concepts?.length) return result.concepts;
    return extractConcepts(result);
  }, [result]);

  const studyPath = useMemo(() => {
    if (result?.study_path?.length) {
      return result.study_path.map((item, index) => ({
        title: item.topic || item.title || `Step ${index + 1}`,
        detail: item.detail || 'Review this topic from the retrieved notes.',
        priority: (item.priority as 'High' | 'Medium' | 'Low') || (index === 0 ? 'High' : 'Medium'),
      }));
    }
    return buildStudyPath(result);
  }, [result]);

  const stats = useMemo(() => {
    const sourceCount = result?.sources?.length || 0;
    return [
      { title: 'Grounding Strength', value: confidenceLabel(result), change: sourceCount ? `${sourceCount} source chunks attached` : 'Awaiting retrieved evidence', icon: ShieldCheck, tone: 'from-cyan-500/15 via-blue-500/10 to-transparent' },
      { title: 'Intent Route', value: result?.route ? routeLabel(result.route) : '—', change: result?.route ? 'Question classified before answering' : 'Route appears after response', icon: Route, tone: 'from-violet-500/15 via-fuchsia-500/10 to-transparent' },
      { title: 'Concepts Found', value: String(concepts.length || '—'), change: concepts.length ? 'Structured revision signal' : 'Concept map appears after answer', icon: Target, tone: 'from-emerald-500/15 via-emerald-500/5 to-transparent' },
    ];
  }, [result, concepts.length]);

  async function handleAsk() {
    if (!question.trim()) {
      setError('Please enter a question.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const payload: Record<string, unknown> = { question: question.trim() };
      if (owner.trim()) payload.owner = owner.trim();
      const data = await postJson<AskResponse>(
  `${backendUrl}/ask`,
  payload
);
      setResult(data);
    } catch (err: any) {
      setError(err?.message || 'Request failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <motion.div {...motionProps} className="px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="grid gap-4 md:grid-cols-3">
          {stats.map((item) => <StatCard key={item.title} {...item} />)}
        </div>

        <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
          <GlassCard className="p-6 sm:p-7">
            <div className="mb-6 flex items-start justify-between gap-4">
              <SectionHeading
                eyebrow="Question Workspace"
                title="Ask your study memory"
                description="Ask a question and Recall will classify the study intent, retrieve the right lecture chunks, and produce a source-backed revision answer."
              />
              <div className="hidden rounded-2xl border border-cyan-400/15 bg-sky-50 px-3 py-2 text-xs text-sky-700 sm:block">
                Endpoint connected
              </div>
            </div>

            <div className="space-y-5">
              <Field label="Question" hint="Natural language supported">
                <PremiumTextarea
                  rows={7}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Example: What should I review before an exam on supervised learning?"
                />
              </Field>

              <div className="grid gap-4 md:grid-cols-2">
                <Field label="Owner" hint="Optional user scope">
                  <PremiumInput value={owner} onChange={(e) => setOwner(e.target.value)} placeholder="nour" />
                </Field>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
  <div className="mb-2 flex items-center gap-2">
    <ShieldCheck className="h-4 w-4 text-cyan-400" />
    <span className="font-medium">Grounded Study Mode</span>
  </div>

  <p className="text-sm text-slate-400">
    Recall retrieves relevant lecture content, explains concepts from
    your materials, and suggests what to review next.
  </p>
</div>
              </div>

              {error ? <StatusBanner tone="warning" title="Request issue" message={error} /> : null}

              <div className="flex flex-wrap gap-3">
                <GradientButton onClick={handleAsk} disabled={loading}>
                  {loading ? 'Studying your notes...' : 'Ask Recall'}
                  {!loading && <ArrowRight className="h-4 w-4" />}
                </GradientButton>
                <GradientButton variant="secondary" onClick={() => setQuestion('What should I revise for supervised learning?')}>Load demo question</GradientButton>
              </div>
            </div>
          </GlassCard>

          <GlassCard className="overflow-hidden p-6 sm:p-7">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(186,230,253,0.55),transparent_28%),radial-gradient(circle_at_bottom_right,rgba(233,213,255,0.45),transparent_28%)]" />
            <div className="relative">
              <SectionHeading eyebrow="Recall Pipeline" title="From question to revision plan" description="Show the intelligence of the RAG system instead of hiding it behind a chat box." />
              <div className="mt-6 grid gap-3">
                {[
                  ['Intent classified', result?.route ? routeLabel(result.route) : 'Waiting for question'],
                  ['Notes retrieved', result?.sources?.length ? `${result.sources.length} source chunks` : 'Sources will appear here'],
                  ['Concepts extracted', concepts.length ? `${concepts.length} review concepts` : 'Concept map generated after answer'],
                  ['Study path generated', studyPath.length ? `${studyPath.length} revision steps` : 'Revision path appears after answer'],
                ].map(([title, detail], index) => (
                  <div key={title} className="flex items-center gap-4 rounded-3xl border border-white/70 bg-white/70 p-4 shadow-sm">
                    <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-400 to-violet-500 text-sm font-semibold text-white">{index + 1}</div>
                    <div>
                      <p className="text-sm font-semibold text-slate-900">{title}</p>
                      <p className="text-xs leading-5 text-slate-500">{detail}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </GlassCard>
        </div>

        <AnimatePresence mode="wait">
          <motion.div key={result ? 'result' : 'empty'} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.28 }}>
            {result ? (
              <div className="space-y-6">
                <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
                  <GlassCard className="p-6 sm:p-7">
                    <SectionHeading eyebrow="Learning Intent" title={routeLabel(result.route)} description={routeDescription(result.route)} />
                    <div className="mt-5 flex flex-wrap gap-3">
                      <FeatureChip icon={Sparkles} label="Intent routing" tone="border-violet-200 bg-violet-50 text-violet-700" />
                      <FeatureChip icon={SearchCheck} label="Source-backed" tone="border-sky-200 bg-sky-50 text-sky-700" />
                      <FeatureChip icon={GraduationCap} label="Revision focused" tone="border-emerald-200 bg-emerald-50 text-emerald-700" />
                    </div>
                  </GlassCard>

                  <GlassCard className="p-6 sm:p-7">
                    <SectionHeading eyebrow="Concept Map" title="Key topics found" description="Recall turns retrieved chunks into a compact revision checklist." />
                    <div className="mt-5 flex flex-wrap gap-3">
                      {concepts.length ? concepts.map((concept) => (
                        <span key={concept} className="rounded-full border border-slate-200 bg-white/80 px-4 py-2 text-sm font-medium text-slate-700">
                          {concept}
                        </span>
                      )) : <p className="text-sm text-slate-500">No concepts extracted yet.</p>}
                    </div>
                  </GlassCard>
                </div>

                <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
                  <GlassCard className="p-6 sm:p-7">
                    <div className="mb-5 flex items-center justify-between gap-4">
                      <SectionHeading eyebrow="Generated Response" title="Grounded answer" description="Written only from retrieved notes and supporting source chunks." />
                      <div className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-700">
                        {confidenceLabel(result)} grounding
                      </div>
                    </div>

                    <div className="rounded-[28px] border border-slate-200 bg-slate-50/80 p-5 text-sm leading-7 text-slate-700 whitespace-pre-wrap">
                      {result.answer || 'No answer returned.'}
                    </div>

                    {result.study_hint ? (
                      <div className="mt-5 rounded-2xl border border-sky-200 bg-sky-50 p-4">
                        <p className="text-sm font-medium text-sky-800">Study Hint</p>
                        <p className="mt-2 text-sm leading-7 text-sky-800">{result.study_hint}</p>
                      </div>
                    ) : null}
                  </GlassCard>

                  <GlassCard className="p-6 sm:p-7">
                    <SectionHeading eyebrow="Revision Path" title="What to study next" description="A visual roadmap generated from the answer and retrieved concepts." />
                    <div className="mt-6 space-y-4">
                      {studyPath.length ? studyPath.map((step, index) => (
                        <div key={`${step.title}-${index}`} className="relative rounded-[26px] border border-slate-200 bg-white/75 p-4">
                          <div className="flex gap-4">
                            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-400 to-violet-500 text-sm font-semibold text-white">{index + 1}</div>
                            <div>
                              <div className="flex flex-wrap items-center gap-2">
                                <p className="font-semibold text-slate-900">{step.title}</p>
                                <span className="rounded-full border border-violet-200 bg-violet-50 px-2 py-1 text-[11px] text-violet-700">{step.priority}</span>
                              </div>
                              <p className="mt-2 text-sm leading-6 text-slate-500">{step.detail}</p>
                            </div>
                          </div>
                        </div>
                      )) : (
                        <p className="rounded-2xl border border-slate-200 bg-white/70 p-4 text-sm text-slate-500">No study path generated yet.</p>
                      )}
                    </div>
                  </GlassCard>
                </div>

                <GlassCard className="p-6 sm:p-7">
                  <div className="mb-5 flex items-center justify-between">
                    <SectionHeading eyebrow="Grounding" title="Sources used" description="Each card explains why the source mattered, not just its vector score." />
                    <div className="rounded-full border border-slate-200 bg-white/85 px-3 py-2 text-xs text-slate-600">
                      {result.sources?.length || 0} sources
                    </div>
                  </div>
                  <div className="grid gap-4 lg:grid-cols-3">
                    {result.sources?.length ? (
                      result.sources.map((source, index) => (
                        <details key={`${source.document_id}-${index}`} className="group overflow-hidden rounded-[26px] border border-slate-200 bg-white/70 open:bg-white/95">
                          <summary className="flex cursor-pointer list-none items-start justify-between gap-4 px-5 py-4">
                            <div>
                              <p className="text-sm font-semibold text-slate-900">{sourceTitle(source, index)}</p>
                              <p className="mt-2 text-xs leading-5 text-slate-500">{sourceReason(source)}</p>
                              <div className="mt-3 flex flex-wrap gap-2">
                                <span className="rounded-full bg-sky-50 px-2 py-1 text-[11px] text-sky-700">Chunk {String(source.chunk_index ?? 'unknown')}</span>
                                <span className="rounded-full bg-emerald-50 px-2 py-1 text-[11px] text-emerald-700">Retrieval ready</span>
                              </div>
                            </div>
                            <ChevronRight className="mt-1 h-4 w-4 shrink-0 text-slate-500 transition group-open:rotate-90" />
                          </summary>
                          <div className="border-t border-slate-200 px-5 py-4 text-sm leading-7 text-slate-600">
                            {source.text || 'No source text returned.'}
                          </div>
                        </details>
                      ))
                    ) : (
                      <div className="rounded-2xl border border-slate-200 bg-white/70 px-4 py-4 text-sm text-slate-500">
                        No sources returned.
                      </div>
                    )}
                  </div>
                </GlassCard>
              </div>
            ) : (
              <GlassCard className="p-8 text-center">
                <BookOpen className="mx-auto h-10 w-10 text-slate-500" />
                <h4 className="mt-4 text-xl font-semibold tracking-tight text-slate-900">Waiting for a study question</h4>
                <p className="mx-auto mt-3 max-w-2xl text-sm leading-7 text-slate-500">
                  Once you submit a question, Recall will show the detected intent, key concepts, grounded answer, revision path, and source evidence.
                </p>
              </GlassCard>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
