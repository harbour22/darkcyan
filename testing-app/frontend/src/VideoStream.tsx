import React, { useEffect, useRef, useState } from "react";
import Sparkline from "./Sparkline";

interface Detection {
  cls: number;
  conf: number;
  xyxy: [number, number, number, number];
}

interface YoloMessage {
  source_id: string;
  frame_count: number;
  fps_video: number;
  fps_yolo: number;
  source_fps: number;
  yolo_ms: number;
  queue_delay_ms: number;
  detections: Detection[];
}

interface Props {
  sourceId: string; // "cam1", "cam2"
}

const HISTORY_LEN = 100;

const VideoStream: React.FC<Props> = ({ sourceId }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectionsRef = useRef<Detection[]>([]);
  const [yoloState, setYoloState] = useState<YoloMessage | null>(null);

  const [fpsVideoHistory, setFpsVideoHistory] = useState<number[]>([]);
  const [fpsYoloHistory, setFpsYoloHistory] = useState<number[]>([]);
  const [yoloMsHistory, setYoloMsHistory] = useState<number[]>([]);
  const [queueDelayHistory, setQueueDelayHistory] = useState<number[]>([]);

  // Video WS (frames)
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws_video/${sourceId}`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log(`Video WS connected for ${sourceId}`);
    };

    ws.onmessage = (event) => {
      if (!(event.data instanceof ArrayBuffer)) return;

      const jpegBytes = new Uint8Array(event.data);
      const blob = new Blob([jpegBytes], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);

      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) {
          URL.revokeObjectURL(url);
          return;
        }
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          URL.revokeObjectURL(url);
          return;
        }

        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;

        ctx.drawImage(img, 0, 0);

        // overlay YOLO detections
        const dets = detectionsRef.current;
        ctx.lineWidth = 2;
        ctx.strokeStyle = "red";
        ctx.font = "12px sans-serif";
        ctx.fillStyle = "rgba(255,0,0,0.4)";

        dets.forEach((det) => {
          const [x1, y1, x2, y2] = det.xyxy;
          const w = x2 - x1;
          const h = y2 - y1;

          ctx.beginPath();
          ctx.rect(x1, y1, w, h);
          ctx.stroke();

          const label = `cls:${det.cls} ${(det.conf * 100).toFixed(1)}%`;
          const textWidth = ctx.measureText(label).width;
          const textHeight = 12;

          ctx.fillRect(x1, y1 - textHeight, textWidth + 4, textHeight + 2);
          ctx.fillStyle = "white";
          ctx.fillText(label, x1 + 2, y1 - 2);
          ctx.fillStyle = "rgba(255,0,0,0.4)";
        });

        URL.revokeObjectURL(url);
      };

      img.src = url;
    };

    ws.onclose = () => {
      console.log(`Video WS disconnected for ${sourceId}`);
    };

    return () => {
      ws.close();
    };
  }, [sourceId]);

  // YOLO WS (detections + metrics)
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws_yolo/${sourceId}`);

    ws.onopen = () => {
      console.log(`YOLO WS connected for ${sourceId}`);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as YoloMessage;
        detectionsRef.current = msg.detections || [];
        setYoloState(msg);

        setFpsVideoHistory((prev) => {
          const next = [...prev, msg.fps_video];
          return next.length > HISTORY_LEN ? next.slice(-HISTORY_LEN) : next;
        });
        setFpsYoloHistory((prev) => {
          const next = [...prev, msg.fps_yolo];
          return next.length > HISTORY_LEN ? next.slice(-HISTORY_LEN) : next;
        });
        setYoloMsHistory((prev) => {
          const next = [...prev, msg.yolo_ms];
          return next.length > HISTORY_LEN ? next.slice(-HISTORY_LEN) : next;
        });
        setQueueDelayHistory((prev) => {
          const next = [...prev, msg.queue_delay_ms];
          return next.length > HISTORY_LEN ? next.slice(-HISTORY_LEN) : next;
        });
      } catch (e) {
        console.error("Failed to parse YOLO message", e);
      }
    };

    ws.onclose = () => {
      console.log(`YOLO WS disconnected for ${sourceId}`);
    };

    return () => {
      ws.close();
    };
  }, [sourceId]);

  return (
    <div style={{ border: "1px solid #ccc", padding: "0.5rem" }}>
      <h3>{sourceId}</h3>
      <canvas ref={canvasRef} style={{ maxWidth: "100%", display: "block" }} />

      {yoloState && (
        <div style={{ fontSize: "0.8rem", marginTop: "0.5rem" }}>
          <div>
            Source FPS: {yoloState.source_fps.toFixed(1)} | Video FPS:{" "}
            {yoloState.fps_video.toFixed(1)} | YOLO FPS:{" "}
            {yoloState.fps_yolo.toFixed(1)}
          </div>
          <div>
            YOLO Latency: {yoloState.yolo_ms.toFixed(1)} ms | Queue Delay:{" "}
            {yoloState.queue_delay_ms.toFixed(1)} ms
          </div>
          <div>Frame Count: {yoloState.frame_count}</div>

          <div style={{ display: "flex", gap: "1rem", marginTop: "0.5rem" }}>
            <Sparkline
              label="Video FPS"
              data={fpsVideoHistory}
              color="#0af"
              decimals={1}
            />
            <Sparkline
              label="YOLO FPS"
              data={fpsYoloHistory}
              color="#fa0"
              decimals={1}
            />
            <Sparkline
              label="YOLO ms"
              data={yoloMsHistory}
              color="#f33"
              decimals={1}
            />
            <Sparkline
              label="Queue ms"
              data={queueDelayHistory}
              color="#6c3"
              decimals={1}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoStream;
