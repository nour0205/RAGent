import { AlertTriangle, CheckCircle2, Clock3 } from 'lucide-react';
import type { StatusTone } from '../../types';
import { cn } from '../../lib/utils';

export default function StatusBanner({ tone, title, message }: { tone: StatusTone; title: string; message: string }) {
  const styles = {
    success: 'border-[#ead6e2] bg-[#fbf7fb] text-[#6f3d7b]',
    warning: 'border-[#eadcc7] bg-[#fffaf4] text-[#7a4e2a]',
    info: 'border-[#ead6e2] bg-[#fbf7fb] text-[#6f3d7b]',
  }[tone];

  const Icon = tone === 'success' ? CheckCircle2 : tone === 'warning' ? AlertTriangle : Clock3;

  return (
    <div className={cn('rounded-2xl border px-4 py-3', styles)}>
      <div className="flex items-start gap-3">
        <Icon className="mt-0.5 h-4 w-4 shrink-0" />
        <div>
          <p className="text-sm font-medium">{title}</p>
          <p className="mt-1 text-sm opacity-90">{message}</p>
        </div>
      </div>
    </div>
  );
}
