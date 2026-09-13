import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function ThreatChart({
  data,
}) {
  return (
    <div
      className="
        h-[280px]
        w-full
      "
    >
      <ResponsiveContainer
        width="100%"
        height="100%"
      >
        <LineChart
          data={data}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#1e293b"
            vertical={false}
          />

          <XAxis
            dataKey="sequence"
            stroke="#64748b"
            fontSize={11}
            tickLine={false}
            axisLine={false}
          />

          <YAxis
            domain={[0, 1]}
            stroke="#64748b"
            fontSize={11}
            tickLine={false}
            axisLine={false}
          />

          <Tooltip
            contentStyle={{
              background: "#0f172a",
              border: "1px solid #1e293b",
              borderRadius: "12px",
              color: "#e2e8f0",
            }}
          />

          <Line
            type="monotone"
            dataKey="threat"
            stroke="#60a5fa"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />

          <Line
            type="monotone"
            dataKey="anomaly"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default ThreatChart;