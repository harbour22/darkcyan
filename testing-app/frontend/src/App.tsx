import React, { useEffect, useState } from "react";
import VideoStream from "./VideoStream";
import Dashboard from "./Dashboard";

interface SourceInfo {
  id: string;
  path: string;
}

interface SourcesResponse {
  sources: SourceInfo[];
}

function App() {
  const [sources, setSources] = useState<SourceInfo[]>([]);

  useEffect(() => {
    const fetchSources = async () => {
      try {
        const res = await fetch("http://localhost:8000/sources");
        const data = (await res.json()) as SourcesResponse;
        setSources(data.sources);
      } catch (e) {
        console.error("Failed to fetch sources", e);
      }
    };

    fetchSources();
  }, []);

  return (
    <div style={{ padding: "1rem", fontFamily: "system-ui, sans-serif" }}>
      <Dashboard />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
          gap: "1rem",
        }}
      >
        {sources.map((s) => (
          <VideoStream key={s.id} sourceId={s.id} />
        ))}
      </div>
    </div>
  );
}

export default App;
