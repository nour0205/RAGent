import { cn } from '../../lib/utils';

export default function FeatureChip({ icon: Icon, label, tone }: { icon: any; label: string; tone: string }) {
  return (
    <div className={cn('signature-badge inline-flex items-center gap-2 rounded-full px-3.5 py-2 text-sm font-semibold', tone)}>
      <Icon className="h-4 w-4" />
      {label}
    </div>
  );
}
