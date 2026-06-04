export function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(' ');
}

export function routeLabel(route?: string) {
  const labels: Record<string, string> = {
    concept_explanation: 'Concept Explanation',
    source_recall: 'Source Recall',
    exam_preparation: 'Exam Preparation',
    unknown: 'Unknown',
  };

  return labels[route || 'unknown'] || (route || 'unknown').replace(/_/g, ' ');
}
