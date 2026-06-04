import { cn } from '../../lib/utils';

export default function FeatureChip({ icon: Icon, label, tone }: { icon: any; label: string; tone: string }) {
  return (
    <div className={cn('inline-flex items-center gap-2 rounded-full border px-3 py-2 text-sm', tone)}>
      <Icon className="h-4 w-4" />
      {label}
    </div>
  );
}
