import asyncio
import threading
import time
import json
from collections import deque
from typing import Optional, Dict, List
import queue

import cv2
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import logging
import torch
from ultralytics import YOLO

import av

YOLO_INPUT_WIDTH = 640      # width YOLO sees
YOLO_MIN_CONF = 0.3         # optional: filter low-confidence boxes

YOLO_MODEL_PATH = "/Users/chris/Documents/developer/darkcyan_data/engines/det/yolov8_4.15_large-det.mlpackage"
YOLO_NUM_WORKERS = 2          # try 2 first; can bump to 3–4 if stable
YOLO_INPUT_WIDTH = 640        # downscaled width for YOLO inference
YOLO_MIN_CONF = 0.3           # filter low-confidence boxes



# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("video_server")

# ---------------- Device ----------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device for YOLO inference: {device}")


# ---------------- Shared State ----------------
class AppState:
    def __init__(self, source_fps: float = 0.0):
        self.lock = threading.Lock()
        self.jpeg_bytes: Optional[bytes] = None
        self.width = 0
        self.height = 0
        self.last_frame_ts = 0.0
        self.frame_count = 0

        self.source_fps = source_fps
        self.video_fps = 0.0
        self.yolo_fps = 0.0
        self.yolo_ms = 0.0
        self.queue_delay_ms = 0.0

        self.detections: List[dict] = []

        # NEW: track timestamps of YOLO-completed frames across all workers
        self._yolo_ts = deque(maxlen=60)  # last ~60 events, adjust as needed

    def set_source_fps(self, fps: float):
        with self.lock:
            self.source_fps = fps

    def update_video_fps(self, fps: float):
        with self.lock:
            self.video_fps = fps

    def update_yolo_fps(self, fps: float):
        with self.lock:
        # you can keep this if other code calls it, but we’ll prefer the new method
            self.yolo_fps = fps

    def record_yolo_frame(self, ts: float):
        """Record the completion time of a YOLO inference and recompute FPS."""
        with self.lock:
            self._yolo_ts.append(ts)
            if len(self._yolo_ts) >= 2:
                elapsed = self._yolo_ts[-1] - self._yolo_ts[0]
                if elapsed > 0:
                    self.yolo_fps = (len(self._yolo_ts) - 1) / elapsed

    def update_metrics(self, yolo_ms: float, queue_delay_ms: float):
        with self.lock:
            self.yolo_ms = yolo_ms
            self.queue_delay_ms = queue_delay_ms

    def update_frame(self, jpeg_bytes: bytes, w: int, h: int, ts: float):
        """Latest raw video frame (no YOLO overlay)."""
        with self.lock:
            self.jpeg_bytes = jpeg_bytes
            self.width = w
            self.height = h
            self.last_frame_ts = ts
            self.frame_count += 1  # count of video frames

    def update_detections(self, dets: List[dict]):
        with self.lock:
            self.detections = dets

    def snapshot(self):
        with self.lock:
            return {
                "width": self.width,
                "height": self.height,
                "fps_video": self.video_fps,
                "fps": self.yolo_fps,  # keep name 'fps' as YOLO FPS for compatibility
                "source_fps": self.source_fps,
                "last_frame_ts": self.last_frame_ts,
                "frame_count": self.frame_count,
                "yolo_ms": self.yolo_ms,
                "queue_delay_ms": self.queue_delay_ms,
                "detections": self.detections,
            }

    def get_latest_frame(self):
        with self.lock:
            return self.jpeg_bytes, self.last_frame_ts


app_states: Dict[str, AppState] = {}
frame_queues: Dict[str, "queue.Queue"] = {}
# Keep handles so we can join during shutdown and exit cleanly.
worker_threads: List[threading.Thread] = []

stop_event = threading.Event()

# ---------------- Video sources ----------------
VIDEO_SOURCES: Dict[str, object] = {
    "cam1": "/Users/chris/Documents/developer/darkcyan_data/test_data/video/Reolink4kFront-20230910-191600.mp4",
    "cam2": "/Users/chris/Documents/developer/darkcyan_data/test_data/video/Reolink4kFront-20230910-191600.mp4",
    "cam3": "/Users/chris/Documents/developer/darkcyan_data/test_data/video/Reolink4kFront-20230910-191600.mp4",
}

def is_camera_source(source) -> bool:
    # int or numeric string → treat as camera index
    if isinstance(source, int):
        return True
    if isinstance(source, str) and source.isdigit():
        return True
    return False

def frame_producer(
    source_id: str,
    source: str,
    frame_queue: "queue.Queue",
    state: AppState,
    stop_event: threading.Event,
    max_width: int = 1024,
):
    """
    Frame producer for file-based video sources using PyAV.

    - Decodes video using ffmpeg (PyAV).
    - Scales frames to max_width using ffmpeg filter graph.
    - Converts frames to yuvj420p for JPEG compatibility.
    - Encodes JPEG using PyAV MJPEG encoder (no OpenCV).
    - Extracts NV12 frames for YOLO workers.
    """

    import av
    import time
    import queue
    import cv2
    from collections import deque

    logger.info(f"[{source_id}] Using PyAV for file decoding: {source}")

    try:
        container = av.open(source)
    except Exception as e:
        logger.error(f"[{source_id}] Failed to open file with PyAV: {e}")
        return

    video_stream = container.streams.video[0]

    # Hint ffmpeg about threading
    try:
        video_stream.thread_type = "AUTO"
    except Exception:
        pass
    try:
        if video_stream.codec_context:
            video_stream.codec_context.thread_count = 2
    except Exception:
        pass

    # Determine FPS
    if video_stream.average_rate:
        video_fps = float(video_stream.average_rate)
    else:
        video_fps = 25.0

    state.set_source_fps(video_fps)
    frame_interval = 1.0 / video_fps

    logger.info(f"[{source_id}] File FPS (from PyAV): {video_fps}")

    # -------------------------
    # Filter graph:
    #   decode -> scale -> format(yuvj420p) -> buffersink
    # -------------------------
    filter_graph = av.filter.Graph()
    buffer_src = filter_graph.add_buffer(template=video_stream)
    scale = filter_graph.add("scale", f"{max_width}:-1")
    fmt = filter_graph.add("format", "yuvj420p")   # crucial for MJPEG
    buffer_sink = filter_graph.add("buffersink")

    buffer_src.link_to(scale)
    scale.link_to(fmt)
    fmt.link_to(buffer_sink)

    filter_graph.configure()

    # We'll compute Video FPS using timestamps
    ts_deque = deque(maxlen=60)
    # Track schedule for pacing so we match the file FPS without cumulative drift.
    next_frame_time = time.time()

    # MJPEG encoder (initialized once we know width/height)
    mjpeg_codec = None

    logger.info(f"[{source_id}] Frame producer (PyAV + MJPEG) started")

    while not stop_event.is_set():
        try:
            for decoded_frame in container.decode(video=0):
                if stop_event.is_set():
                    break

                # Push into filter graph
                buffer_src.push(decoded_frame)

                while True:
                    try:
                        filt_frame = buffer_sink.pull()
                    except Exception:
                        break  # no more frames from this packet

                    h = filt_frame.height
                    w = filt_frame.width
                    ts = time.time()

                    # -------------------------
                    # Lazy init MJPEG encoder
                    # -------------------------
                    if mjpeg_codec is None:
                        mjpeg_codec = av.CodecContext.create("mjpeg", "w")
                        mjpeg_codec.width = w
                        mjpeg_codec.height = h
                        mjpeg_codec.pix_fmt = "yuvj420p"   # REQUIRED for JPEG
                        logger.info(
                            f"[{source_id}] MJPEG encoder init: {w}x{h}, pix_fmt=yuvj420p"
                        )

                    # -------------------------
                    # Encode JPEG via ffmpeg
                    # -------------------------
                    jpeg = None
                    for packet in mjpeg_codec.encode(filt_frame):
                        jpeg = bytes(packet)
                        break

                    if jpeg:
                        state.update_frame(jpeg, w, h, ts)

                        ts_deque.append(ts)
                        if len(ts_deque) >= 2:
                            elapsed = ts_deque[-1] - ts_deque[0]
                            if elapsed > 0:
                                state.update_video_fps((len(ts_deque)-1)/elapsed)

                    # -------------------------
                    # Provide NV12 frame to YOLO worker
                    # -------------------------
                    nv12 = filt_frame.to_ndarray(format="nv12")

                    try:
                        frame_queue.put((nv12, w, h, ts), block=False)
                    except queue.Full:
                        # Drop oldest
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            frame_queue.put((nv12, w, h, ts), block=False)
                        except queue.Full:
                            pass

                    # -------------------------
                    # Pace based on source FPS
                    # -------------------------
                    # Use accumulated schedule instead of previous-iteration timing.
                    next_frame_time += frame_interval
                    sleep_time = next_frame_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        # Fell behind – reset schedule to now to avoid long-term drift.
                        next_frame_time = time.time()

            # EOF → loop file
            if stop_event.is_set():
                break

            logger.info(f"[{source_id}] EOF reached, seeking(0)")
            container.seek(0)

        except Exception as e:
            logger.error(f"[{source_id}] Error during PyAV decode: {e}")
            time.sleep(0.1)

    container.close()
    logger.info(f"[{source_id}] Frame producer stopped")



def yolo_worker(
    source_id: str,
    frame_queue: "queue.Queue",
    state: AppState,
    stop_event: threading.Event,
    model_path: str,
    worker_idx: int,
):
    """YOLO worker: consumes NV12 frames, converts to BGR, downscales for YOLO, etc."""
    logger.info(f"[{source_id}][yolo{worker_idx}] Initializing YOLO model on {device}")
    local_model = YOLO(model_path, task='detect')

    logger.info(f"[{source_id}][yolo{worker_idx}] YOLO worker started")

    while not stop_event.is_set():
        try:
            nv12, w, h, ts_in = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        queue_delay_ms = (time.time() - ts_in) * 1000.0

        # NV12 -> BGR (CPU; clean place to swap for hardware conversion later)
        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

        # Downscale for YOLO speed
        yolo_input = bgr
        scale_x = scale_y = 1.0
        if w > YOLO_INPUT_WIDTH:
            new_w = YOLO_INPUT_WIDTH
            new_h = int(h * (YOLO_INPUT_WIDTH / w))
            yolo_input = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale_x = w / new_w
            scale_y = h / new_h

        # YOLO inference
        start = time.time()
        results = local_model(yolo_input, device=device, verbose=False)
        yolo_ms = (time.time() - start) * 1000.0

        # Extract detections, scale boxes back to original size
        dets: List[dict] = []
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                try:
                    xyxy = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < YOLO_MIN_CONF:
                        continue

                    x1, y1, x2, y2 = xyxy
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y

                    dets.append({"cls": cls_id, "conf": conf, "xyxy": [x1, y1, x2, y2]})
                except Exception:
                    continue

        # Update shared state
        state.update_detections(dets)
        state.update_metrics(yolo_ms, queue_delay_ms)
        state.record_yolo_frame(time.time())

    logger.info(f"[{source_id}][yolo{worker_idx}] YOLO worker stopped")

def startup():
    logger.info("Startup: initializing video pipelines")

    for sid, src in VIDEO_SOURCES.items():
        state = AppState()
        app_states[sid] = state

        q: "queue.Queue" = queue.Queue(maxsize=1)
        frame_queues[sid] = q

        # FPS probe
        if is_camera_source(src):
            cap_tmp = cv2.VideoCapture(int(src))
            fps_tmp = cap_tmp.get(cv2.CAP_PROP_FPS) or 30
            cap_tmp.release()
        else:
            try:
                container_tmp = av.open(src)
                stream_tmp = container_tmp.streams.video[0]
                fps_tmp = float(stream_tmp.average_rate) if stream_tmp.average_rate else 25.0
                container_tmp.close()
            except Exception:
                fps_tmp = 25.0

        state.set_source_fps(fps_tmp)

        # frame producer
        producer_thread = threading.Thread(
            target=frame_producer,
            args=(sid, src, q, state, stop_event),
            daemon=True,
        )
        producer_thread.start()
        worker_threads.append(producer_thread)

        # multiple YOLO workers
        for worker_idx in range(YOLO_NUM_WORKERS):
            worker_thread = threading.Thread(
                target=yolo_worker,
                args=(sid, q, state, stop_event, YOLO_MODEL_PATH, worker_idx),
                daemon=True,
            )
            worker_thread.start()
            worker_threads.append(worker_thread)

        logger.info(f"[{sid}] Producer + {YOLO_NUM_WORKERS} YOLO workers started")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    # Run blocking startup logic without blocking event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, startup)

    yield  # <-- execution yields to the running server

    # Shutdown logic
    stop_event.set()
    logger.info("Shutdown: stop_event set")

    # Join worker threads so shutdown finishes cleanly.
    for t in worker_threads:
        if t.is_alive():
            t.join(timeout=2.0)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- State & Health endpoints ----------------
@app.get("/state/{source_id}")
def get_state(source_id: str):
    if source_id not in app_states:
        raise HTTPException(status_code=404, detail="Unknown source_id")
    snap = app_states[source_id].snapshot()
    return JSONResponse(snap)


@app.get("/health")
def health():
    data = {}
    now = time.time()
    for sid, st in app_states.items():
        snap = st.snapshot()
        last_ts = snap["last_frame_ts"]
        data[sid] = {
            "alive": last_ts > 0,
            "last_frame_age": (now - last_ts) if last_ts else None,
            "source_fps": snap["source_fps"],
            "video_fps": snap["fps_video"],
            "yolo_fps": snap["fps"],
            "frame_count": snap["frame_count"],
            "yolo_ms": snap["yolo_ms"],
            "queue_delay_ms": snap["queue_delay_ms"],
        }
    return JSONResponse(data)

@app.get("/sources")
def list_sources():
    # basic metadata – can be extended later
    return JSONResponse(
        {
            "sources": [
                {
                    "id": sid,
                    "path": str(src),
                }
                for sid, src in VIDEO_SOURCES.items()
            ]
        }
    )


# ---------------- WebSockets ----------------

# 1) Video stream: raw JPEG frames at capture rate (no YOLO overlay)
@app.websocket("/ws_video/{source_id}")
async def ws_video(ws: WebSocket, source_id: str):
    if source_id not in app_states:
        await ws.close(code=1008)
        return

    state = app_states[source_id]
    await ws.accept()
    last_ts = 0.0
    logger.info(f"[{source_id}] Video WebSocket connected")

    try:
        while True:
            jpeg, ts = state.get_latest_frame()
            if jpeg is not None and ts != last_ts:
                await ws.send_bytes(jpeg)
                last_ts = ts
            else:
                await asyncio.sleep(0.005)
    except Exception as e:
        logger.info(f"[{source_id}] Video WebSocket disconnected: {e}")


# 2) YOLO detections: JSON metadata at YOLO rate
@app.websocket("/ws_yolo/{source_id}")
async def ws_yolo(ws: WebSocket, source_id: str):
    if source_id not in app_states:
        await ws.close(code=1008)
        return

    state = app_states[source_id]
    await ws.accept()
    last_frame_count = -1
    logger.info(f"[{source_id}] YOLO WebSocket connected")

    try:
        while True:
            snap = state.snapshot()
            fc = snap["frame_count"]
            if fc != last_frame_count:
                payload = json.dumps(
                    {
                        "source_id": source_id,
                        "frame_count": fc,
                        "fps_video": snap["fps_video"],
                        "fps_yolo": snap["fps"],
                        "source_fps": snap["source_fps"],
                        "yolo_ms": snap["yolo_ms"],
                        "queue_delay_ms": snap["queue_delay_ms"],
                        "detections": snap["detections"],
                    }
                )
                await ws.send_text(payload)
                last_frame_count = fc
            else:
                await asyncio.sleep(0.05)
    except Exception as e:
        logger.info(f"[{source_id}] YOLO WebSocket disconnected: {e}")
