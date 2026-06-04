export default function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div>
      <p className="text-xs uppercase tracking-[0.28em] text-slate-400">{eyebrow}</p>
      <h3 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">{title}</h3>
      <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">{description}</p>
    </div>
  );
}
