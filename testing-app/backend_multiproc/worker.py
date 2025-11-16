from __future__ import annotations

import multiprocessing as mp
import time
from collections import deque
from pathlib import Path
from typing import List

import av
import cv2
import torch
from ultralytics import YOLO

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parent
    import sys

    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.append(str(PACKAGE_ROOT))

    from config import (
        YOLO_MODEL_PATH,
        YOLO_INPUT_WIDTH,
        YOLO_MIN_CONF,
        JPEG_QUALITY,
        DISPLAY_MAX_WIDTH,
    )
    from messages import FramePacket, ShutdownNotice
else:
    from .config import (
        YOLO_MODEL_PATH,
        YOLO_INPUT_WIDTH,
        YOLO_MIN_CONF,
        JPEG_QUALITY,
        DISPLAY_MAX_WIDTH,
    )
    from .messages import FramePacket, ShutdownNotice


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def worker_main(
    source_id: str,
    source_path: str,
    out_queue: mp.Queue,
    stop_event: mp.Event,
    source_fps: float | None = None,
):
    """Decode video + run YOLO entirely inside this process."""
    logger = mp.get_logger()
    device = _device()
    logger.info("[%s] worker starting on %s", source_id, _device())

    model = YOLO(str(YOLO_MODEL_PATH), task="detect")

    container = av.open(source_path)
    video_stream = container.streams.video[0]

    if source_fps is None:
        if video_stream.average_rate:
            fps = float(video_stream.average_rate)
        else:
            fps = 25.0
    else:
        fps = source_fps
    frame_interval = 1.0 / fps

    filter_graph = av.filter.Graph()
    buffer_src = filter_graph.add_buffer(template=video_stream)
    scale = filter_graph.add("scale", f"{DISPLAY_MAX_WIDTH}:-1")
    fmt = filter_graph.add("format", "bgr24")
    buffer_sink = filter_graph.add("buffersink")

    buffer_src.link_to(scale)
    scale.link_to(fmt)
    fmt.link_to(buffer_sink)
    filter_graph.configure()

    ts_deque = deque(maxlen=60)
    next_frame_time = time.time()
    video_frame_count = 0
    yolo_ts = deque(maxlen=60)

    try:
        while not stop_event.is_set():
            for decoded in container.decode(video=0):
                if stop_event.is_set():
                    break
                buffer_src.push(decoded)

                while True:
                    try:
                        frame = buffer_sink.pull()
                    except Exception:
                        break

                    ts = time.time()
                    h, w = frame.height, frame.width
                    bgr = frame.to_ndarray(format="bgr24")

                    yolo_frame = bgr
                    if bgr.shape[1] > YOLO_INPUT_WIDTH:
                        new_w = YOLO_INPUT_WIDTH
                        new_h = int(bgr.shape[0] * (YOLO_INPUT_WIDTH / bgr.shape[1]))
                        yolo_frame = cv2.resize(
                            bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
                        )
                        scale_x = bgr.shape[1] / new_w
                        scale_y = bgr.shape[0] / new_h
                    else:
                        scale_x = scale_y = 1.0

                    success, encoded = cv2.imencode(
                        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    )
                    if not success:
                        continue

                    # YOLO inference
                    start = time.time()
                    results = model(yolo_frame, device=device, verbose=False)
                    yolo_ms = (time.time() - start) * 1000.0

                    dets: List[dict] = []
                    res = results[0]
                    if hasattr(res, "boxes") and res.boxes is not None:
                        for box in res.boxes:
                            try:
                                xyxy = box.xyxy[0].tolist()
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                if conf < YOLO_MIN_CONF:
                                    continue
                                x1, y1, x2, y2 = xyxy
                                dets.append(
                                    {
                                        "cls": cls,
                                        "conf": conf,
                                        "xyxy": [
                                            x1 * scale_x,
                                            y1 * scale_y,
                                            x2 * scale_x,
                                            y2 * scale_y,
                                        ],
                                    }
                                )
                            except Exception:
                                continue

                    ts_deque.append(ts)
                    video_frame_count += 1
                    fps_video = 0.0
                    if len(ts_deque) >= 2:
                        elapsed = ts_deque[-1] - ts_deque[0]
                        if elapsed > 0:
                            fps_video = (len(ts_deque) - 1) / elapsed

                    yolo_ts.append(time.time())
                    yolo_fps = 0.0
                    if len(yolo_ts) >= 2:
                        elapsed = yolo_ts[-1] - yolo_ts[0]
                        if elapsed > 0:
                            yolo_fps = (len(yolo_ts) - 1) / elapsed

                    packet = FramePacket(
                        source_id=source_id,
                        timestamp=ts,
                        width=w,
                        height=h,
                        jpeg=encoded.tobytes(),
                        video_fps=fps_video,
                        yolo_fps=yolo_fps,
                        yolo_ms=yolo_ms,
                        queue_delay_ms=(time.time() - ts) * 1000.0,
                        detections=dets,
                        source_fps=fps,
                    )

                    _emit(out_queue, packet)

                    next_frame_time += frame_interval
                    sleep_time = max(0.0, next_frame_time - time.time())
                    if sleep_time:
                        time.sleep(sleep_time)

                if stop_event.is_set():
                    break

            container.seek(0)

    finally:
        container.close()
        _emit(out_queue, ShutdownNotice(source_id))
        logger.info("[%s] worker exiting", source_id)


def _emit(queue: mp.Queue, payload):
    """Attempt non-blocking send, dropping oldest if needed."""
    while True:
        try:
            queue.put(payload, timeout=0.5)
            break
        except Exception:
            try:
                _ = queue.get_nowait()
            except Exception:
                break
