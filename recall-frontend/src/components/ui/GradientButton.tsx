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
      whileHover={{ y: -2 }}
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'group relative inline-flex items-center justify-center gap-2 rounded-full px-5 py-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50',
        variant === 'primary'
          ? 'bg-[#0b0a12] text-white shadow-[0_18px_38px_rgba(55,38,62,0.18)] hover:bg-[#211724]'
          : 'border border-[#eee7ef] bg-white text-[#0b0a12] shadow-sm hover:border-[#e4d5e7] hover:bg-[#fbf7fb]',
        className,
      )}
    >
      <span className="relative inline-flex items-center gap-2">{children}</span>
    </motion.button>
  );
}
