function MetricCard({
  title,
  value,
  subtitle,
  icon,
  accent = "blue",
}) {
  const accentClasses = {
    blue: "text-blue-400 bg-blue-500/10 border-blue-500/20",
    green:
      "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    yellow:
      "text-amber-400 bg-amber-500/10 border-amber-500/20",
    red:
      "text-red-400 bg-red-500/10 border-red-500/20",
    purple:
      "text-violet-400 bg-violet-500/10 border-violet-500/20",
  };

  return (
    <div
      className="
        rounded-2xl
        border
        border-slate-800
        bg-slate-950/70
        p-5
        shadow-lg
        shadow-black/10
      "
    >
      <div
        className="
          mb-4
          flex
          items-center
          justify-between
        "
      >
        <p
          className="
            text-sm
            font-medium
            text-slate-400
          "
        >
          {title}
        </p>

        <div
          className={`
            rounded-xl
            border
            p-2.5
            ${accentClasses[accent]}
          `}
        >
          {icon}
        </div>
      </div>

      <div
        className="
          text-3xl
          font-semibold
          tracking-tight
          text-white
        "
      >
        {value}
      </div>

      <p
        className="
          mt-2
          text-xs
          leading-relaxed
          text-slate-500
        "
      >
        {subtitle}
      </p>
    </div>
  );
}

export default MetricCard;