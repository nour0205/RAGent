import { motion } from 'framer-motion';
import { ArrowRight, BookOpen, ShieldCheck, Target, Search, Upload, Database } from 'lucide-react';
import type { Page } from '../types';
import { motionProps } from '../lib/constants';
import GlassCard from '../components/ui/GlassCard';
import FeatureChip from '../components/ui/FeatureChip';
import GradientButton from '../components/ui/GradientButton';
import ChoiceCard from '../components/ui/ChoiceCard';

export default function LandingPage({ setPage }: { setPage: (p: Page) => void }) {
  return (
    <motion.div {...motionProps} className="px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-8">
        <GlassCard className="overflow-hidden px-6 py-8 sm:px-8 lg:px-10 lg:py-12">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(186,230,253,0.55),transparent_26%),radial-gradient(circle_at_center_right,rgba(233,213,255,0.38),transparent_24%),radial-gradient(circle_at_bottom_left,rgba(167,243,208,0.28),transparent_20%)]" />
          <div className="relative grid gap-8 lg:grid-cols-[1.3fr_0.7fr] lg:items-center">
            <div>
              <div className="mb-5 flex flex-wrap gap-3">
                <FeatureChip icon={ShieldCheck} label="Grounded Answers" tone="border-sky-200 bg-sky-50 text-sky-700" />
                <FeatureChip icon={Target} label="Intent Routing" tone="border-violet-200 bg-violet-50 text-violet-700" />
                <FeatureChip icon={BookOpen} label="Study Memory" tone="border-emerald-200 bg-emerald-50 text-emerald-700" />
              </div>
              <h1 className="max-w-4xl text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl lg:text-6xl">
                Study from your own notes.
              </h1>
              <p className="mt-5 max-w-2xl text-base leading-8 text-slate-600 sm:text-lg">
                Recall retrieves lecture content, explains concepts using grounded sources, and guides what to revise next.
              </p>
              <div className="mt-8 flex flex-wrap gap-4">
                <GradientButton onClick={() => setPage('ask')}>Launch Ask Recall <ArrowRight className="h-4 w-4" /></GradientButton>
                <GradientButton variant="secondary" onClick={() => setPage('ingest')}>Open Ingestion Flow</GradientButton>
              </div>
            </div>

            <GlassCard className="p-5">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500">Retrieval Snapshot</p>
                  <p className="text-2xl font-semibold tracking-tight text-slate-900">Recall Flow</p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white/80 px-3 py-2 text-xs text-slate-500">Source-backed study</div>
              </div>

              <div className="grid gap-4">
                {[
                  ['Notes Retrieved', 'Hybrid search', '82%', 'from-sky-400 via-blue-500 to-violet-400', 'text-sky-700'],
                  ['Answer Grounding', 'Sources attached', '74%', 'from-emerald-400 to-sky-400', 'text-emerald-700'],
                  ['Revision Guidance', 'Study path ready', '91%', 'from-violet-400 to-fuchsia-400', 'text-violet-700'],
                ].map(([label, state, width, gradient, stateColor]) => (
                  <div key={label} className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                    <div className="mb-3 flex items-center justify-between text-sm">
                      <span className="text-slate-500">{label}</span>
                      <span className={stateColor}>{state}</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-slate-200">
                      <motion.div initial={{ width: 0 }} animate={{ width }} transition={{ duration: 1 }} className={`h-full rounded-full bg-gradient-to-r ${gradient}`} />
                    </div>
                  </div>
                ))}
              </div>
            </GlassCard>
          </div>
        </GlassCard>

        <div className="grid gap-6 lg:grid-cols-3">
          <ChoiceCard
            title="Ask Recall"
            description="Ask a question and see the detected intent, key concepts, grounded answer, revision path, and source evidence."
            icon={Search}
            onClick={() => setPage('ask')}
            tone="from-sky-100 via-sky-50 to-transparent"
          />
          <ChoiceCard
            title="Ingest Notes"
            description="Upload or paste study material with course, source, owner, and topic tags through a clean multi-step note ingestion flow."
            icon={Upload}
            onClick={() => setPage('ingest')}
            tone="from-violet-100 via-fuchsia-50 to-transparent"
          />
          <ChoiceCard
            title="Knowledge Base"
            description="Browse your indexed notes as retrieval-ready study assets with previews, chunks, and source status."
            icon={Database}
            onClick={() => setPage('knowledge')}
            tone="from-emerald-100 via-sky-50 to-transparent"
          />
        </div>
      </div>
    </motion.div>
  );
}
