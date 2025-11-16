import React from "react";

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  label: string;
  decimals?: number;
}

const Sparkline: React.FC<SparklineProps> = ({
  data,
  width = 110,
  height = 50,
  color = "#0af",
  label,
  decimals = 1,
}) => {
  if (data.length === 0) {
    return (
      <div style={{ fontSize: "0.75rem" }}>
        {label}: n/a
      </div>
    );
  }

  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1 || 1)) * (width - 2) + 1;
    const norm = (v - min) / range;
    const y = height - 1 - norm * (height - 2);
    return `${x},${y}`;
  });

  const latest = data[data.length - 1];

  return (
    <div style={{ fontSize: "0.75rem" }}>
      <div>{label}: {latest.toFixed(decimals)}</div>
      <svg width={width} height={height}>
        <polyline
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          points={points.join(" ")}
        />
      </svg>
    </div>
  );
};

export default Sparkline;
