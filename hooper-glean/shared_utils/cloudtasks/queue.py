from dataclasses import dataclass
from enum import Enum


class QueueType(Enum):
    """Different types of processing queues.

    VIDEO_PROCESSING - Used for CPU operations on full video files.
    VIDEO_COMPRESSION - Used for FFMPEG operations on full video files.
    CLIP_INFERENCE - Used for GPU inference on video clips.
    HOOP_DETECTION - Used for GPU object detection on video frames.
    APP_NOTIFICATION - Used for sending delayed notifications to users.
    COMBINE - Used for Hooper's Combine application (CPU only).
    """

    VIDEO_PROCESSING = "video-processing-queue"
    VIDEO_COMPRESSION = "video-compression-queue"
    CLIP_INFERENCE = "clip-inference-queue"
    HOOP_DETECTION = "hoop-detection-queue"
    APP_NOTIFICATION = "app-notification-queue"
    COMBINE = "combine-queue"


@dataclass
class QueueConfig:
    name: str
    max_dispatches_per_second: float
    max_concurrent_dispatches: int
    max_attempts: int
    min_backoff_seconds: int = 5
    max_backoff_seconds: int = 60
    max_retry_duration_seconds: int = 300


# Backoff seconds = how long cloud tasks wait before retrying a failed task
# Max retry duration seconds = total time limit for all tretries
QUEUE_CONFIGS = {
    QueueType.VIDEO_PROCESSING: QueueConfig(
        name=QueueType.VIDEO_PROCESSING.value,
        max_dispatches_per_second=5.0,
        max_concurrent_dispatches=4,
        max_attempts=1,
    ),
    QueueType.VIDEO_COMPRESSION: QueueConfig(
        name=QueueType.VIDEO_COMPRESSION.value,
        max_dispatches_per_second=5.0,
        max_concurrent_dispatches=4,
        max_attempts=3,
        min_backoff_seconds=60,
        max_backoff_seconds=600,
        max_retry_duration_seconds=3600,
    ),
    QueueType.CLIP_INFERENCE: QueueConfig(
        name=QueueType.CLIP_INFERENCE.value,
        max_dispatches_per_second=5.0,
        max_concurrent_dispatches=16,
        max_attempts=3,
        min_backoff_seconds=30,
        max_backoff_seconds=300,
        max_retry_duration_seconds=600,
    ),
    QueueType.HOOP_DETECTION: QueueConfig(
        name=QueueType.HOOP_DETECTION.value,
        max_dispatches_per_second=5.0,
        max_concurrent_dispatches=16,
        max_attempts=3,
        min_backoff_seconds=30,
        max_backoff_seconds=300,
        max_retry_duration_seconds=600,
    ),
    QueueType.APP_NOTIFICATION: QueueConfig(
        name=QueueType.APP_NOTIFICATION.value,
        max_dispatches_per_second=50.0,
        max_concurrent_dispatches=16,
        max_attempts=3,
        min_backoff_seconds=1,
        max_backoff_seconds=10,
        max_retry_duration_seconds=300,
    ),
    QueueType.COMBINE: QueueConfig(
        name=QueueType.COMBINE.value,
        max_dispatches_per_second=5.0,
        max_concurrent_dispatches=4,
        max_attempts=3,
        min_backoff_seconds=30,
        max_backoff_seconds=300,
        max_retry_duration_seconds=1800,
    ),
}

QUEUE_DEVICE_MAPPING = {
    QueueType.VIDEO_PROCESSING: "cpu",
    QueueType.VIDEO_COMPRESSION: "cpu",
    QueueType.CLIP_INFERENCE: "gpu",
    QueueType.HOOP_DETECTION: "gpu",
    QueueType.APP_NOTIFICATION: "cpu",
    QueueType.COMBINE: "cpu",
}
