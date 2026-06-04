import type { ReactNode } from 'react';
import { Brain, ChevronRight, Search, Sparkles, Upload, Database } from 'lucide-react';
import type { Page } from '../../types';
import { cn } from '../../lib/utils';

export default function AppShell({
  children,
  page,
  setPage,
  backendUrl,
}: {
  children: ReactNode;
  page: Page;
  setPage: (p: Page) => void;
  backendUrl: string;
}) {
  const nav = [
    { key: 'ask', label: 'Ask Recall', icon: Search, tone: 'from-sky-100 via-white to-transparent' },
    { key: 'ingest', label: 'Ingest Notes', icon: Upload, tone: 'from-violet-100 via-white to-transparent' },
    { key: 'knowledge', label: 'Knowledge Base', icon: Database, tone: 'from-emerald-100 via-white to-transparent' },
  ] as const;

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,rgba(186,230,253,0.5),transparent_22%),radial-gradient(circle_at_top_right,rgba(233,213,255,0.4),transparent_24%),linear-gradient(180deg,#fbfdff_0%,#f7f9fc_45%,#f5f7fb_100%)] text-slate-900">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute inset-0 bg-[linear-gradient(to_bottom,rgba(255,255,255,0.28),rgba(248,250,252,0.78))]" />
      </div>

      <div className="relative flex min-h-screen">
        <aside className="hidden w-[290px] shrink-0 border-r border-slate-200/80 bg-white/65 backdrop-blur-2xl lg:flex lg:flex-col">
          <div className="p-6">
            <div className="rounded-3xl border border-white/90 bg-white/75 p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)] backdrop-blur-xl">
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-100 via-violet-100 to-fuchsia-100 shadow-sm">
                  <Brain className="h-6 w-6 text-sky-700" />
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Study Memory</p>
                  <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Recall</h1>
                </div>
              </div>
              <p className="mt-4 text-sm leading-6 text-slate-600">
                A calm workspace for revising concepts, tracing sources, and preparing from your own notes.
              </p>
            </div>
          </div>

          <nav className="flex-1 space-y-2 px-4">
            <button
              onClick={() => setPage('home')}
              className={cn(
                'group relative w-full overflow-hidden rounded-3xl border px-4 py-4 text-left transition-all',
                page === 'home'
                  ? 'border-slate-200 bg-white/90 shadow-[0_14px_40px_rgba(15,23,42,0.06)]'
                  : 'border-transparent bg-white/45 hover:bg-white/72',
              )}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-slate-100/80 to-transparent" />
              <div className="relative flex items-center gap-3">
                <div className="rounded-2xl bg-slate-100 p-2"><Sparkles className="h-5 w-5 text-slate-700" /></div>
                <div>
                  <p className="font-medium text-slate-900">Overview</p>
                  <p className="text-sm text-slate-500">Launchpad and product flow</p>
                </div>
              </div>
            </button>

            {nav.map((item) => {
              const Icon = item.icon;
              const active = page === item.key;
              return (
                <button
                  key={item.key}
                  onClick={() => setPage(item.key)}
                  className={cn(
                    'group relative w-full overflow-hidden rounded-3xl border px-4 py-4 text-left transition-all',
                    active
                      ? 'border-slate-200 bg-white/92 shadow-[0_14px_40px_rgba(15,23,42,0.06)]'
                      : 'border-transparent bg-white/45 hover:-translate-y-0.5 hover:bg-white/72',
                  )}
                >
                  <div className={cn('absolute inset-0 bg-gradient-to-r opacity-100', item.tone)} />
                  <div className="relative flex items-center gap-3">
                    <div className="rounded-2xl bg-white/85 p-2 backdrop-blur-xl">
                      <Icon className="h-5 w-5 text-slate-700" />
                    </div>
                    <div className="min-w-0">
                      <p className="font-medium text-slate-900">{item.label}</p>
                      <p className="truncate text-sm text-slate-500">Focused workspace</p>
                    </div>
                    <ChevronRight className="ml-auto h-4 w-4 text-slate-400 transition-transform group-hover:translate-x-0.5" />
                  </div>
                </button>
              );
            })}
          </nav>

          <div className="p-4">
            <div className="rounded-[28px] border border-emerald-200/90 bg-white/75 p-5 shadow-[0_20px_60px_rgba(15,23,42,0.05)] backdrop-blur-xl">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-900">System Status</p>
                  <p className="text-xs text-slate-500">Backend connection target</p>
                </div>
                <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs text-emerald-700">
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                    <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
                  </span>
                  Online Ready
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50/90 px-3 py-3 text-xs text-slate-600">
                {backendUrl}
              </div>
            </div>
          </div>
        </aside>

        <main className="flex-1 overflow-hidden">{children}</main>
      </div>
    </div>
  );
}
