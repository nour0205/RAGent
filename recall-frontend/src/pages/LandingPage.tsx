import { motion } from 'framer-motion';
import { ArrowRight, Database, Search, Upload } from 'lucide-react';
import type { Page } from '../types';
import { motionProps } from '../lib/constants';
import GlassCard from '../components/ui/GlassCard';
import GradientButton from '../components/ui/GradientButton';
import ChoiceCard from '../components/ui/ChoiceCard';

export default function LandingPage({ setPage }: { setPage: (p: Page) => void }) {
  return (
    <motion.div {...motionProps} className="px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1200px] space-y-6">
        <GlassCard className="soft-rose-panel overflow-hidden px-7 py-10 sm:px-10 lg:px-12 lg:py-14">
          <div className="relative max-w-4xl">
            <p className="section-kicker">Study brief</p>
            <h1 className="mt-5 max-w-4xl text-4xl font-semibold tracking-[-0.055em] text-[#0b0a12] sm:text-5xl lg:text-6xl">
              Ask your notes. Get a grounded answer.
            </h1>
            <p className="mt-5 max-w-2xl text-base leading-8 text-[#6f6878]">
              Recall gives you a clear study brief with clickable evidence from your own material.
            </p>
            <div className="mt-7 flex flex-wrap gap-4">
              <GradientButton onClick={() => setPage('ask')}>Ask Recall <ArrowRight className="h-4 w-4" /></GradientButton>
              <GradientButton variant="secondary" onClick={() => setPage('ingest')}>Add notes</GradientButton>
            </div>
          </div>
        </GlassCard>

        <div className="grid gap-5 lg:grid-cols-3">
          <ChoiceCard
            title="Ask"
            description="Get a clear answer shaped from your indexed material."
            icon={Search}
            onClick={() => setPage('ask')}
          />
          <ChoiceCard
            title="Add Notes"
            description="Paste material once and make it searchable for future study."
            icon={Upload}
            onClick={() => setPage('ingest')}
          />
          <ChoiceCard
            title="Library"
            description="Browse the notes Recall can use as source evidence."
            icon={Database}
            onClick={() => setPage('knowledge')}
          />
        </div>
      </div>
    </motion.div>
  );
}
