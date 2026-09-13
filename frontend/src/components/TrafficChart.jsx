import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function TrafficChart({
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
            dataKey="packetRate"
            stroke="#34d399"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default TrafficChart;