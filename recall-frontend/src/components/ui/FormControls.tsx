import type { InputHTMLAttributes, ReactNode, TextareaHTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block space-y-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium text-slate-700">{label}</span>
        {hint ? <span className="text-xs text-slate-400">{hint}</span> : null}
      </div>
      {children}
    </label>
  );
}

export function PremiumInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        'w-full rounded-2xl border border-slate-200 bg-white/90 px-4 py-3 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-sky-300 focus:bg-white',
        props.className,
      )}
    />
  );
}

export function PremiumTextarea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...props}
      className={cn(
        'w-full rounded-3xl border border-slate-200 bg-white/90 px-4 py-4 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-sky-300 focus:bg-white',
        props.className,
      )}
    />
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between rounded-2xl border border-slate-200 bg-white/90 px-4 py-3"
    >
      <span className="text-sm text-slate-700">{label}</span>
      <span className={cn('flex h-7 w-12 rounded-full p-1 transition', checked ? 'bg-sky-500' : 'bg-slate-200')}>
        <span className={cn('h-5 w-5 rounded-full bg-white transition', checked ? 'translate-x-5' : 'translate-x-0')} />
      </span>
    </button>
  );
}
