from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parent
    import sys

    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.append(str(PACKAGE_ROOT))

    from config import VIDEO_SOURCES
    from messages import FramePacket, ShutdownNotice
    from state import SupervisorRegistry
    from worker import worker_main
else:
    from .config import VIDEO_SOURCES
    from .messages import FramePacket, ShutdownNotice
    from .state import SupervisorRegistry
    from .worker import worker_main


try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass


@dataclass
class WorkerHandle:
    process: mp.Process
    queue: mp.Queue
    stop_event: mp.Event
    pump_thread: threading.Thread
    pump_stop: threading.Event


class Supervisor:
    def __init__(self):
        self.registry = SupervisorRegistry()
        self.workers: Dict[str, WorkerHandle] = {}

    def start_workers(self):
        for source_id, source_path in VIDEO_SOURCES.items():
            if source_id in self.workers:
                continue

            result_queue: mp.Queue = mp.Queue(maxsize=4)
            stop_event = mp.Event()

            process = mp.Process(
                target=worker_main,
                args=(source_id, source_path, result_queue, stop_event),
                daemon=True,
            )
            process.start()

            pump_stop = threading.Event()
            pump_thread = threading.Thread(
                target=self._pump_worker_queue,
                args=(source_id, result_queue, pump_stop),
                daemon=True,
            )
            pump_thread.start()

            self.workers[source_id] = WorkerHandle(
                process=process,
                queue=result_queue,
                stop_event=stop_event,
                pump_thread=pump_thread,
                pump_stop=pump_stop,
            )

    def stop_workers(self):
        for source_id, handle in self.workers.items():
            handle.stop_event.set()
        for source_id, handle in self.workers.items():
            if handle.process.is_alive():
                handle.process.join(timeout=5.0)
            handle.pump_stop.set()
        for handle in self.workers.values():
            handle.pump_thread.join(timeout=2.0)
        self.workers.clear()

    def _pump_worker_queue(
        self, source_id: str, result_queue: mp.Queue, stop_flag: threading.Event
    ):
        state = self.registry.ensure(source_id)
        while not stop_flag.is_set():
            try:
                payload = result_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if isinstance(payload, FramePacket):
                state.update_from_packet(payload)
            elif isinstance(payload, ShutdownNotice):
                break

    def get_state(self, source_id: str):
        if source_id not in self.registry.states:
            raise HTTPException(status_code=404, detail="Unknown source_id")
        return self.registry.states[source_id].snapshot()


supervisor = Supervisor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, supervisor.start_workers)
    yield
    await loop.run_in_executor(None, supervisor.stop_workers)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/sources")
def list_sources():
    return JSONResponse({"sources": [{"id": sid, "path": path} for sid, path in VIDEO_SOURCES.items()]})


@app.get("/state/{source_id}")
def state_endpoint(source_id: str):
    data = supervisor.get_state(source_id)
    return JSONResponse(data)


@app.get("/health")
def health():
    now = time.time()
    payload = {}
    for sid, st in supervisor.registry.states.items():
        snap = st.snapshot()
        last_ts = snap["last_frame_ts"]
        payload[sid] = {
            "alive": last_ts > 0 and (now - last_ts) < 5.0,
            "last_frame_age": (now - last_ts) if last_ts else None,
            "video_fps": snap["fps_video"],
            "yolo_fps": snap["fps_yolo"],
            "frame_count": snap["frame_count"],
            "source_fps": snap["source_fps"],
            "yolo_ms": snap["yolo_ms"],
            "queue_delay_ms": snap["queue_delay_ms"],
        }
    return JSONResponse(payload)


@app.websocket("/ws_video/{source_id}")
async def ws_video(ws: WebSocket, source_id: str):
    if source_id not in supervisor.registry.states:
        supervisor.registry.ensure(source_id)
    await ws.accept()
    last_ts = 0.0
    state = supervisor.registry.states[source_id]
    try:
        while True:
            frame, ts = state.latest_frame()
            if frame is not None and ts != last_ts:
                await ws.send_bytes(frame)
                last_ts = ts
            else:
                await asyncio.sleep(0.01)
    except Exception:
        pass


@app.websocket("/ws_yolo/{source_id}")
async def ws_yolo(ws: WebSocket, source_id: str):
    if source_id not in supervisor.registry.states:
        supervisor.registry.ensure(source_id)
    await ws.accept()
    last_frame = -1
    state = supervisor.registry.states[source_id]
    try:
        while True:
            snap = state.snapshot()
            if snap["frame_count"] != last_frame:
                await ws.send_json(
                    {
                        "source_id": source_id,
                        "frame_count": snap["frame_count"],
                        "fps_video": snap["fps_video"],
                        "fps_yolo": snap["fps_yolo"],
                        "source_fps": snap["source_fps"],
                        "yolo_ms": snap["yolo_ms"],
                        "queue_delay_ms": snap["queue_delay_ms"],
                        "detections": snap["detections"],
                    }
                )
                last_frame = snap["frame_count"]
            else:
                await asyncio.sleep(0.05)
    except Exception:
        pass
