import type { ReactNode } from 'react';
import { cn } from '../../lib/utils';

export default function GlassCard({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cn('relative overflow-hidden rounded-3xl border border-white/90 bg-white/72 shadow-[0_20px_60px_rgba(15,23,42,0.06)] backdrop-blur-xl', className)}>
      <div className="absolute inset-0 bg-gradient-to-br from-white/90 via-white/45 to-transparent" />
      <div className="relative">{children}</div>
    </div>
  );
}
