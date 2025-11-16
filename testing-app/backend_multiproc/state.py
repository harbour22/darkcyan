import threading
import time
from collections import deque
from typing import Dict, List, Optional

from .messages import FramePacket


class SourceState:
    """Parent-process view of worker output."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg: Optional[bytes] = None
        self.width = 0
        self.height = 0
        self.last_frame_ts = 0.0
        self.frame_count = 0
        self.video_fps = 0.0
        self.yolo_fps = 0.0
        self.yolo_ms = 0.0
        self.queue_delay_ms = 0.0
        self.detections: List[dict] = []
        self.source_fps = 0.0
        self._yolo_ts = deque(maxlen=60)

    def update_from_packet(self, packet: FramePacket):
        with self.lock:
            self.jpeg = packet.jpeg
            self.width = packet.width
            self.height = packet.height
            self.last_frame_ts = packet.timestamp
            self.frame_count += 1
            self.video_fps = packet.video_fps
            self.yolo_fps = packet.yolo_fps
            self.yolo_ms = packet.yolo_ms
            self.queue_delay_ms = packet.queue_delay_ms
            self.detections = packet.detections
            self.source_fps = packet.source_fps
            self._yolo_ts.append(packet.timestamp)

    def snapshot(self):
        with self.lock:
            return {
                "width": self.width,
                "height": self.height,
                "frame_count": self.frame_count,
                "last_frame_ts": self.last_frame_ts,
                "source_fps": self.source_fps,
                "fps_video": self.video_fps,
                "fps_yolo": self.yolo_fps,
                "yolo_ms": self.yolo_ms,
                "queue_delay_ms": self.queue_delay_ms,
                "detections": self.detections,
            }

    def latest_frame(self):
        with self.lock:
            return self.jpeg, self.last_frame_ts


class SupervisorRegistry:
    def __init__(self):
        self.states: Dict[str, SourceState] = {}

    def ensure(self, source_id: str) -> SourceState:
        if source_id not in self.states:
            self.states[source_id] = SourceState()
        return self.states[source_id]

    def snapshot(self):
        return {sid: state.snapshot() for sid, state in self.states.items()}
