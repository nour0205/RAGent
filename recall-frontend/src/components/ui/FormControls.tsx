import type { InputHTMLAttributes, ReactNode, TextareaHTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block space-y-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-semibold text-[#27212d]">{label}</span>
        {hint ? <span className="text-xs font-medium text-[#8a8090]">{hint}</span> : null}
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
        'w-full rounded-[18px] border border-[#eee7ef] bg-white px-4 py-3 text-sm text-[#0b0a12] outline-none transition placeholder:text-[#aaa0ae] focus:border-[#d96f9f] focus:bg-white focus:shadow-[0_0_0_4px_rgba(217,111,159,0.10)]',
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
        'w-full rounded-[26px] border border-[#eee7ef] bg-white px-4 py-4 text-sm leading-7 text-[#0b0a12] outline-none transition placeholder:text-[#aaa0ae] focus:border-[#d96f9f] focus:bg-white focus:shadow-[0_0_0_4px_rgba(217,111,159,0.10)]',
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
      className="flex w-full items-center justify-between rounded-[18px] border border-[#eee7ef] bg-white px-4 py-3 text-sm text-[#0b0a12]"
    >
      <span>{label}</span>
      <span className={cn('h-7 w-12 rounded-full p-1 transition', checked ? 'bg-[#d96f9f]' : 'bg-[#eee7ef]')}>
        <span className={cn('block h-5 w-5 rounded-full bg-white transition', checked ? 'translate-x-5' : 'translate-x-0')} />
      </span>
    </button>
  );
}
