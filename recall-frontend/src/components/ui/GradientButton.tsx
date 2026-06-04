import type { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '../../lib/utils';

export default function GradientButton({
  children,
  onClick,
  variant = 'primary',
  type = 'button',
  disabled = false,
  className = '',
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary';
  type?: 'button' | 'submit';
  disabled?: boolean;
  className?: string;
}) {
  return (
    <motion.button
      whileTap={{ scale: 0.985 }}
      whileHover={{ y: -1 }}
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-2xl px-5 py-3 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50',
        variant === 'primary'
          ? 'border border-sky-200 bg-gradient-to-r from-sky-500 via-blue-500 to-violet-500 text-white shadow-[0_18px_45px_rgba(59,130,246,0.18)]'
          : 'border border-slate-200 bg-white/85 text-slate-700 hover:bg-white',
        className,
      )}
    >
      {children}
    </motion.button>
  );
}
