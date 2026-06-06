import type { ReactNode } from 'react';
import { cn } from '../../lib/utils';

export default function GlassCard({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cn('recall-card paper-grain', className)}>
      <div className="relative">{children}</div>
    </div>
  );
}
