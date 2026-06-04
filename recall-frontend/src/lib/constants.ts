export const trendData = [
  { name: 'Mon', value: 22 },
  { name: 'Tue', value: 31 },
  { name: 'Wed', value: 29 },
  { name: 'Thu', value: 44 },
  { name: 'Fri', value: 38 },
  { name: 'Sat', value: 52 },
  { name: 'Sun', value: 49 },
];

export const readinessData = [
  { name: 'Readiness', value: 78, fill: 'url(#readinessFill)' },
];

export const motionProps = {
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.45, ease: 'easeOut' as const },
};
