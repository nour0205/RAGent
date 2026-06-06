import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';

export default function ChoiceCard({
  title,
  description,
  icon: Icon,
  onClick,
}: {
  title: string;
  description: string;
  icon: any;
  onClick: () => void;
  tone?: string;
}) {
  return (
    <motion.button
      whileHover={{ y: -4 }}
      whileTap={{ scale: 0.99 }}
      onClick={onClick}
      className="group recall-card relative min-h-[190px] p-6 text-left"
    >
      <div className="relative flex h-full flex-col">
        <div className="mb-5 inline-flex w-fit rounded-[18px] border border-[#eee7ef] bg-[#fbf7fb] p-3 text-[#b85f8b] shadow-sm">
          <Icon className="h-6 w-6" />
        </div>
        <h4 className="text-xl font-bold tracking-tight text-[#0b0a12]">{title}</h4>
        <p className="mt-3 text-sm leading-6 text-[#6f6878]">{description}</p>
        <div className="mt-auto flex items-center gap-2 pt-6 text-sm font-semibold text-[#0b0a12]">
          Open <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
        </div>
      </div>
    </motion.button>
  );
}
