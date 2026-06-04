import { motion } from 'framer-motion';
import GlassCard from './GlassCard';
import { cn } from '../../lib/utils';

export default function StatCard({
  title,
  value,
  change,
  icon: Icon,
  tone,
}: {
  title: string;
  value: string;
  change: string;
  icon: any;
  tone: string;
}) {
  return (
    <motion.div whileHover={{ y: -4 }} className="h-full">
      <GlassCard className="h-full p-5">
        <div className={cn('absolute inset-0 bg-gradient-to-br opacity-100', tone)} />
        <div className="relative flex items-start justify-between gap-4">
          <div>
            <p className="text-sm text-slate-500">{title}</p>
            <p className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">{value}</p>
            <p className="mt-2 text-xs text-slate-500">{change}</p>
          </div>
          <div className="rounded-2xl border border-white/90 bg-white/90 p-3">
            <Icon className="h-5 w-5 text-slate-700" />
          </div>
        </div>
      </GlassCard>
    </motion.div>
  );
}
