# Multiprocess Video + YOLO Supervisor

This prototype splits every camera feed + YOLO pipeline into its own **process** so the Python interpreter lock is no longer a bottleneck. The parent process runs FastAPI, supervises workers, and exposes the same `/state`, `/health`, and WebSocket APIs as the threaded version.

## Layout

| File | Purpose |
| --- | --- |
| `config.py` | Shared constants: YOLO model path, input width, video sources, JPEG quality. |
| `messages.py` | Lightweight dataclasses passed over multiprocessing queues. |
| `worker.py` | Child-process entry point. Decodes frames with PyAV, emits JPEG bytes + YOLO detections. |
| `state.py` | Thread-safe state cache inside the supervisor. |
| `supervisor.py` | FastAPI app that spawns workers, consumes their queues, and serves HTTP/WebSocket endpoints. |

## Running

```bash
cd testing-app
uvicorn backend_multiproc.supervisor:app --reload
```

Each configured camera in `config.py::VIDEO_SOURCES` gets its own process. Shutdown uses FastAPI's lifespan hook to signal every worker, join their processes, and stop the queue pump threads so `Ctrl+C` is clean.
