import type { AskResponse, Source } from '../types';
import { routeLabel } from './utils';

const STOP_WORDS = new Set([
  'the','and','for','with','from','your','this','that','into','then','they','them','were','where','what','when','which','while','about','before','after','using','used','uses','helps','review','revise','study','notes','material','concept','concepts','model','models','data','learning','understanding','important','main','things','first','finally','focus','should','would','could','also','their','there','these','those','because','between','difference','differences'
]);

const KNOWN_CONCEPTS = [
  'supervised learning', 'classification', 'regression', 'overfitting', 'underfitting',
  'bias variance tradeoff', 'bias-variance tradeoff', 'cross validation', 'k-fold cross-validation',
  'normalization', 'functional dependency', 'database transaction', 'acid properties',
  'atomicity', 'consistency', 'isolation', 'durability'
];

function titleCase(value: string) {
  return value
    .replace(/[-_]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function extractConcepts(result: AskResponse | null): string[] {
  if (!result) return [];
  const text = `${result.answer || ''} ${result.study_hint || ''} ${(result.sources || []).map((s) => `${s.document_id || ''} ${s.text || ''}`).join(' ')}`.toLowerCase();
  const concepts: string[] = [];

  for (const concept of KNOWN_CONCEPTS) {
    if (text.includes(concept) && !concepts.some((c) => c.toLowerCase() === concept)) {
      concepts.push(titleCase(concept));
    }
  }

  const boldMatches = `${result.answer || ''} ${result.study_hint || ''}`.match(/\*\*([^*]{3,60})\*\*/g) || [];
  for (const match of boldMatches) {
    const clean = match.replace(/\*\*/g, '').replace(/[:.]/g, '').trim();
    if (clean && !concepts.includes(clean)) concepts.push(clean);
  }

  return concepts.slice(0, 6);
}

export function buildStudyPath(result: AskResponse | null): Array<{ title: string; detail: string; priority: 'High' | 'Medium' | 'Low' }> {
  const concepts = extractConcepts(result);
  if (!result || !concepts.length) return [];

  return concepts.slice(0, 5).map((concept, index) => ({
    title: concept,
    priority: index === 0 ? 'High' : index < 3 ? 'Medium' : 'Low',
    detail: index === 0
      ? 'Start here to anchor the rest of the revision.'
      : index === concepts.length - 1
        ? 'Use this to check whether you can apply the idea.'
        : 'Review this next because it connects to the retrieved sources.',
  }));
}

export function sourceTitle(source: Source, index: number) {
  const raw = source.document_id || `Source ${index + 1}`;
  return titleCase(String(raw));
}

export function sourceReason(source: Source) {
  const text = `${source.text || ''}`.trim();
  if (!text) return 'Supporting passage retrieved from your indexed notes.';
  const firstSentence = text.split(/[.!?]/).find((part) => part.trim().length > 25)?.trim();
  return firstSentence ? `${firstSentence.slice(0, 145)}${firstSentence.length > 145 ? '…' : ''}` : 'Supporting passage retrieved from your indexed notes.';
}

export function confidenceLabel(result: AskResponse | null) {
  const count = result?.sources?.length || 0;
  if (!result?.answer) return 'Waiting';
  if (count >= 3) return 'Strong';
  if (count > 0) return 'Moderate';
  return 'Low';
}

export function routeDescription(route?: string) {
  const label = routeLabel(route);
  const descriptions: Record<string, string> = {
    'Exam Preparation': 'Recall prioritized review-worthy concepts and produced a revision-oriented answer.',
    'Concept Explanation': 'Recall focused on explaining the concept clearly from your notes.',
    'Source Recall': 'Recall focused on locating where the idea appears in your material.',
    Unknown: 'Recall used the fallback route because the study intent was unclear.',
  };
  return descriptions[label] || 'Recall selected the most relevant study route for this question.';
}
