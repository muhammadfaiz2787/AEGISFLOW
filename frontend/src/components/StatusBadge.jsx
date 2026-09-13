function StatusBadge({
  active,
  activeText = "ACTIVE",
  inactiveText = "OFFLINE",
}) {
  return (
    <div
      className={`
        inline-flex
        items-center
        gap-2
        rounded-full
        border
        px-3
        py-1.5
        text-xs
        font-semibold
        ${
          active
            ? `
              border-emerald-500/20
              bg-emerald-500/10
              text-emerald-400
            `
            : `
              border-slate-700
              bg-slate-800/60
              text-slate-400
            `
        }
      `}
    >
      <span
        className={`
          h-2
          w-2
          rounded-full
          ${
            active
              ? "bg-emerald-400 animate-pulse"
              : "bg-slate-500"
          }
        `}
      />

      {active
        ? activeText
        : inactiveText}
    </div>
  );
}

export default StatusBadge;