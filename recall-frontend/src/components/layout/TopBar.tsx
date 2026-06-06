import { Sparkles } from 'lucide-react';
import type { Page } from '../../types';

export default function TopBar({ page }: { page: Page }) {
  const heading = {
    home: { title: 'Recall', desc: 'Study from your notes with answers you can verify and revise.' },
    ask: { title: 'Study Brief', desc: 'Ask once. Get the answer, the evidence, and the next step.' },
    ingest: { title: 'Add Notes', desc: 'Paste study material and make it searchable.' },
    knowledge: { title: 'Library', desc: 'Browse the notes Recall can use as evidence.' },
  }[page];

  return (
    <div className="sticky top-0 z-20 border-b border-[#eee7ef] bg-white/78 backdrop-blur-xl">
      <div className="mx-auto max-w-[1360px] px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.34em] text-[#b85f8b]">
            <Sparkles className="h-3.5 w-3.5" />
            Study from your own notes
          </div>
          <h2 className="text-2xl font-semibold tracking-tight text-[#0b0a12]">{heading.title}</h2>
          <p className="text-sm text-[#6f6878]">{heading.desc}</p>
        </div>
      </div>
    </div>
  );
}
