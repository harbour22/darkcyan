from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FramePacket:
    source_id: str
    timestamp: float
    width: int
    height: int
    jpeg: bytes
    video_fps: float
    yolo_fps: float
    yolo_ms: float
    queue_delay_ms: float
    detections: List[Dict[str, Any]]
    source_fps: float


@dataclass
class Heartbeat:
    source_id: str
    timestamp: float


@dataclass
class ShutdownNotice:
    source_id: str
