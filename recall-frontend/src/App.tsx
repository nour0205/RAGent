import { AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import type { Page } from './types';
import { getDefaultBackendUrl } from './lib/api';
import AppShell from './components/layout/AppShell';
import TopBar from './components/layout/TopBar';
import LandingPage from './pages/LandingPage';
import AskPage from './pages/AskPage';
import IngestPage from './pages/IngestPage';
import KnowledgePage from './pages/KnowledgePage';

export default function App() {
  const [page, setPage] = useState<Page>('home');
  const [backendUrl, setBackendUrl] = useState(getDefaultBackendUrl());

  return (
    <AppShell page={page} setPage={setPage} backendUrl={backendUrl}>
      <TopBar page={page} backendUrl={backendUrl} setBackendUrl={setBackendUrl} />
      <AnimatePresence mode="wait">
        {page === 'home' && <LandingPage key="home" setPage={setPage} />}
        {page === 'ask' && <AskPage key="ask" backendUrl={backendUrl} />}
        {page === 'ingest' && <IngestPage key="ingest" backendUrl={backendUrl} />}
        {page === 'knowledge' && <KnowledgePage key="knowledge" backendUrl={backendUrl} />}
      </AnimatePresence>
    </AppShell>
  );
}
