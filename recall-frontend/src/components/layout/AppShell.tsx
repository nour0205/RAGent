import type { ReactNode } from 'react';
import { Search, Upload, Database } from 'lucide-react';
import type { Page } from '../../types';
import { cn } from '../../lib/utils';

const recallMark = new URL('../../assets/recall-mark.png', import.meta.url).href;

export default function AppShell({
  children,
  page,
  setPage,
}: {
  children: ReactNode;
  page: Page;
  setPage: (p: Page) => void;
}) {
  const nav = [
    { key: 'ask', label: 'Study Brief', desc: 'Ask + verify', icon: Search },
    { key: 'ingest', label: 'Add Notes', desc: 'Paste material', icon: Upload },
    { key: 'knowledge', label: 'Library', desc: 'Study sources', icon: Database },
  ] as const;

  return (
    <div className="min-h-screen text-[#0b0a12]">
      <div className="relative flex min-h-screen">
        <aside className="hidden w-[220px] shrink-0 border-r border-[#eee7ef] bg-white/82 backdrop-blur-xl lg:flex lg:flex-col">
          <div className="px-4 pt-5 pb-7">
            <button
              onClick={() => setPage('home')}
              className="group w-full rounded-[26px] border border-[#eee7ef] bg-white/95 px-3.5 py-3.5 text-left shadow-[0_16px_45px_rgba(55,38,62,0.055)] transition hover:-translate-y-0.5 hover:shadow-[0_22px_60px_rgba(55,38,62,0.075)]"
              aria-label="Go to Recall home"
            >
              <div className="flex items-center gap-3.5">
                <div className="flex h-[52px] w-[52px] shrink-0 items-center justify-center rounded-[18px] bg-[#fff7fb] ring-1 ring-[#ead7e2]">
                  <img
                    src={recallMark}
                    alt=""
                    className="h-11 w-11 object-contain"
                  />
                </div>

                <h1 className="text-[31px] font-extrabold leading-none tracking-[-0.065em] text-[#0b0a12]">
                  Recall
                </h1>
              </div>
            </button>
          </div>

          <nav className="flex-1 space-y-2 px-3">
            {nav.map((item) => (
              <NavButton
                key={item.key}
                active={page === item.key}
                onClick={() => setPage(item.key)}
                icon={item.icon}
                label={item.label}
                desc={item.desc}
              />
            ))}
          </nav>
        </aside>

        <main className="relative z-10 flex-1 overflow-hidden">{children}</main>
      </div>
    </div>
  );
}

function NavButton({ active, onClick, icon: Icon, label, desc }: { active: boolean; onClick: () => void; icon: any; label: string; desc: string }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'group w-full rounded-[20px] border px-3.5 py-3 text-left transition-all',
        active
          ? 'border-[#e5d7e8] bg-[#fbf7fb] text-[#0b0a12] shadow-[0_14px_34px_rgba(55,38,62,0.06)]'
          : 'border-transparent bg-transparent text-[#514858] hover:border-[#eee7ef] hover:bg-white',
      )}
    >
      <div className="flex items-center gap-3">
        <div className={cn('rounded-[15px] p-2 transition', active ? 'bg-white text-[#b85f8b]' : 'bg-[#fbf7fb] text-[#6f6878] group-hover:text-[#b85f8b]')}>
          <Icon className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <p className="font-semibold text-[#0b0a12]">{label}</p>
          <p className="truncate text-sm text-[#7b7280]">{desc}</p>
        </div>
        
      </div>
    </button>
  );
}
