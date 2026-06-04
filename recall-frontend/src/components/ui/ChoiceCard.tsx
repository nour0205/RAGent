import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { cn } from '../../lib/utils';

export default function ChoiceCard({
  title,
  description,
  icon: Icon,
  onClick,
  tone,
}: {
  title: string;
  description: string;
  icon: any;
  onClick: () => void;
  tone: string;
}) {
  return (
    <motion.button
      whileHover={{ y: -6 }}
      whileTap={{ scale: 0.99 }}
      onClick={onClick}
      className="group relative overflow-hidden rounded-[32px] border border-white/90 bg-white/75 p-6 text-left shadow-[0_20px_60px_rgba(15,23,42,0.06)] backdrop-blur-xl"
    >
      <div className={cn('absolute inset-0 bg-gradient-to-br', tone)} />
      <div className="relative">
        <div className="mb-5 inline-flex rounded-2xl border border-white/90 bg-white/85 p-3">
          <Icon className="h-6 w-6 text-slate-700" />
        </div>
        <h4 className="text-xl font-semibold tracking-tight text-slate-900">{title}</h4>
        <p className="mt-3 text-sm leading-6 text-slate-600">{description}</p>
        <div className="mt-6 flex items-center gap-2 text-sm text-slate-800">
          Open workspace <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
        </div>
      </div>
    </motion.button>
  );
}
