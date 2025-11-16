import React, { useEffect, useState } from "react";

interface HealthEntry {
  alive: boolean;
  last_frame_age: number | null;
  source_fps: number;
  video_fps: number;
  yolo_fps: number;
  frame_count: number;
  yolo_ms: number;
  queue_delay_ms: number;
}

type HealthResponse = Record<string, HealthEntry>;

const Dashboard: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch("http://localhost:8000/health");
        const data = (await res.json()) as HealthResponse;
        setHealth(data);
        setError(null);
      } catch (e) {
        setError("Failed to fetch /health "+e);
      }
    };

    fetchHealth();
    const id = setInterval(fetchHealth, 1000); // refresh every second
    return () => clearInterval(id);
  }, []);

  const entries = Object.entries(health);

  return (
    <div style={{ marginBottom: "1rem" }}>
      <h2>System Health</h2>
      {error && <div style={{ color: "red" }}>{error}</div>}
      {entries.length === 0 ? (
        <div style={{ fontSize: "0.85rem" }}>No sources yet…</div>
      ) : (
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "0.8rem",
          }}
        >
          <thead>
            <tr>
              <th style={{ borderBottom: "1px solid #ccc", textAlign: "left" }}>Source</th>
              <th style={{ borderBottom: "1px solid #ccc" }}>Alive</th>
              <th style={{ borderBottom: "1px solid #ccc" }}>Last Frame Age (ms)</th>
              <th style={{ borderBottom: "1px solid #ccc" }}>Source FPS</th>
              <th style={{ borderBottom: "1px solid #ccc" }}>Video FPS</th>
              <th style={{ borderBottom: "1px solid #ccc" }}>YOLO FPS</th>
              <th style={{ borderBottom: "1px solid #ccc" }}>YOLO ms</th>
              <th style={{ borderBottom: "1px solid #ccc" }}>Queue ms</th>
              <th style={{ borderBottom: "1px solid #ccc" }}>Frames</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([id, h]) => (
              <tr key={id}>
                <td style={{ borderBottom: "1px solid #eee" }}>{id}</td>
                <td
                  style={{
                    borderBottom: "1px solid #eee",
                    color: h.alive ? "green" : "red",
                    textAlign: "center",
                  }}
                >
                  ●
                </td>
                <td style={{ borderBottom: "1px solid #eee", textAlign: "right" }}>
                  {h.last_frame_age != null
                    ? (h.last_frame_age * 1000).toFixed(0)
                    : "n/a"}
                </td>
                <td style={{ borderBottom: "1px solid #eee", textAlign: "right" }}>
                  {h.source_fps.toFixed(1)}
                </td>
                <td style={{ borderBottom: "1px solid #eee", textAlign: "right" }}>
                  {h.video_fps.toFixed(1)}
                </td>
                <td style={{ borderBottom: "1px solid #eee", textAlign: "right" }}>
                  {h.yolo_fps.toFixed(1)}
                </td>
                <td style={{ borderBottom: "1px solid #eee", textAlign: "right" }}>
                  {h.yolo_ms.toFixed(1)}
                </td>
                <td style={{ borderBottom: "1px solid #eee", textAlign: "right" }}>
                  {h.queue_delay_ms.toFixed(1)}
                </td>
                <td style={{ borderBottom: "1px solid #eee", textAlign: "right" }}>
                  {h.frame_count}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default Dashboard;
