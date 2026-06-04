import { Server, Sparkles } from 'lucide-react';
import type { Page } from '../../types';

export default function TopBar({
  page,
  backendUrl,
  setBackendUrl,
}: {
  page: Page;
  backendUrl: string;
  setBackendUrl: (v: string) => void;
}) {
  const heading = {
    home: { title: 'Recall Workspace', desc: 'A lighter, calmer interface for studying from your own materials.' },
    ask: { title: 'Ask Recall', desc: 'Grounded answers, intent-aware guidance, and source-backed recall.' },
    ingest: { title: 'Ingest Notes', desc: 'Add course material with clean metadata and structured study context.' },
    knowledge: { title: 'Knowledge Base', desc: 'Browse your notes corpus in a focused, readable layout.' },
  }[page];

  return (
    <div className="sticky top-0 z-20 border-b border-slate-200/80 bg-white/60 backdrop-blur-2xl">
      <div className="flex flex-col gap-4 px-4 py-4 sm:px-6 lg:px-8 xl:flex-row xl:items-center xl:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.28em] text-slate-400">
            <Sparkles className="h-3.5 w-3.5 text-sky-500" />
            Modern Study Interface
          </div>
          <h2 className="text-3xl font-semibold tracking-tight text-slate-900">{heading.title}</h2>
          <p className="mt-1 text-sm text-slate-500">{heading.desc}</p>
        </div>

        <div className="flex w-full flex-col gap-3 sm:flex-row xl:w-auto">
          <div className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white/80 px-4 py-3 backdrop-blur-xl">
            <Server className="h-4 w-4 text-sky-600" />
            <input
              value={backendUrl}
              onChange={(e) => setBackendUrl(e.target.value)}
              className="w-full bg-transparent text-sm text-slate-900 outline-none placeholder:text-slate-400 sm:w-[320px]"
              placeholder="Backend URL"
            />
          </div>
          <button className="rounded-2xl border border-slate-200 bg-white/80 px-4 py-3 text-sm text-slate-700 transition hover:bg-white">
            Calm Focus Mode
          </button>
        </div>
      </div>
    </div>
  );
}
