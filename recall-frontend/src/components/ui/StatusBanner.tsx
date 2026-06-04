import { AlertTriangle, CheckCircle2, Clock3 } from 'lucide-react';
import type { StatusTone } from '../../types';
import { cn } from '../../lib/utils';

export default function StatusBanner({ tone, title, message }: { tone: StatusTone; title: string; message: string }) {
  const styles = {
    success: 'border-emerald-200 bg-emerald-50 text-emerald-800',
    warning: 'border-amber-200 bg-amber-50 text-amber-800',
    info: 'border-sky-200 bg-sky-50 text-sky-800',
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
