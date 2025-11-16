import json
import time
import secrets
from typing import Dict, Any, Optional, List

from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2
from dataclasses import dataclass

from .queue import QUEUE_CONFIGS, QUEUE_DEVICE_MAPPING, QueueType
from ..logger import get_logger
from ..poll import PollManager, TaskStatus, TaskRecord

logger = get_logger(__name__)


@dataclass
class CloudTasksConfig:
    project_id: str
    location: str
    # Deployed cloud run service for CPU tasks
    cloudrun_cpu_service_url: Optional[str] = None
    cloudrun_cpu_invoker_sa: Optional[str] = None
    # Deployed cloud run service for GPU tasks
    cloudrun_gpu_service_url: Optional[str] = None
    cloudrun_gpu_invoker_sa: Optional[str] = None


class CloudTasksManager:
    """Manager for Cloud Tasks queues.

    Args:
        config: Cloud Tasks configuration
    """

    def __init__(self, config: CloudTasksConfig):
        self.config = config
        self.client = tasks_v2.CloudTasksClient()

        # Create queue paths for each queue type
        self.queue_paths = {
            queue_type: self.client.queue_path(
                config.project_id, config.location, queue_config.name
            )
            for queue_type, queue_config in QUEUE_CONFIGS.items()
        }

        # For creating Firebase records for tasks
        self.poll_manager = PollManager()

    def pause_queue(self, queue_type: QueueType) -> bool:
        queue_path = self.queue_paths[queue_type]
        try:
            self.client.pause_queue(request={"name": queue_path})
            logger.error(f"pause_queue - Paused queue {queue_path}")
            return True
        except Exception as e:
            logger.error(f"pause_queue - Failed to pause queue {queue_path}: {e}")
            return False

    def resume_queue(self, queue_type: QueueType) -> bool:
        queue_path = self.queue_paths[queue_type]
        try:
            self.client.resume_queue(request={"name": queue_path})
            logger.error(f"resume_queue - Resumed queue {queue_path}")
            return True
        except Exception as e:
            logger.error(f"resume_queue - Failed to resume queue {queue_path}: {e}")
            return False

    def purge_queue(self, queue_type: QueueType) -> bool:
        queue_path = self.queue_paths[queue_type]
        try:
            self.client.purge_queue(request={"name": queue_path})
            logger.error(f"purge_queue - Purged queue {queue_path}")
            return True
        except Exception as e:
            logger.error(f"purge_queue - Failed to purge queue {queue_path}: {e}")
            return False

    def get_task_records_from_queue(
        self,
        queue_type: QueueType,
        page: int = 0,
        limit: int = 100,
    ) -> List[TaskRecord]:
        """Fetch tasks from a queue.

        Args:
            queue_type: Used to select queue name
            page: Used to specify offset
            limit: Maximum number to return

        Return:
            tasks: Most recent <limit> tasks from the queue.
        """
        return self.poll_manager.get_task_records_from_queue(queue_type, page=page, limit=limit)

    def get_task_record(self, task_id: str) -> Optional[TaskRecord]:
        """Fetch the task receipt from firebase, if it exists.

        Args:
            task_id: Identifier for the query task

        Return:
            task: TaskRecord object or None
        """
        return self.poll_manager.get_task_record(task_id)

    def create_task(
        self,
        queue_type: QueueType,
        endpoint: str,
        payload: Dict[str, Any],
        delay_seconds: int = 0,
        task_id: Optional[str] = None,
    ) -> str:
        """Create a task in the specified queue

        Args:
            queue_type: Which queue to use
            endpoint: Cloud Run endpoint (e.g., "/session/process")
            payload: JSON payload to send
            delay_seconds: Delay before execution
            task_id: Optional custom task ID

        Returns:
            task_name: Full name of created task
        """
        # Some queue types require GPU
        device = QUEUE_DEVICE_MAPPING.get(queue_type)

        if device not in ["cpu", "gpu"]:
            logger.error(f"create_task - Invalid device type: {device}")
            raise ValueError(
                "Invalid device type. Supported types are 'cpu' and 'gpu'."
            )

        if delay_seconds < 0:
            logger.error(f"create_task - Invalid delay_seconds: {delay_seconds}")
            raise ValueError("delay_seconds must be non-negative.")

        # Choose service details based on device type
        cloudrun_service_url = (
            self.config.cloudrun_cpu_service_url
            if device == "cpu"
            else self.config.cloudrun_gpu_service_url
        )
        service_account_email = (
            self.config.cloudrun_cpu_invoker_sa
            if device == "cpu"
            else self.config.cloudrun_gpu_invoker_sa
        )

        if cloudrun_service_url is None:
            logger.error("create_task - Cloud Run service URL is not defined")
            raise ValueError("create_task - Cloud Run service URL is not defined")

        if service_account_email is None:
            logger.error("create_task - Service account email is not defined")
            raise ValueError("create_task - Service account email is not defined")

        try:
            queue_path = self.queue_paths[queue_type]

            task = tasks_v2.Task()

            # Create HTTP request
            task.http_request = tasks_v2.HttpRequest()
            task.http_request.http_method = tasks_v2.HttpMethod.POST
            task.http_request.url = f"{cloudrun_service_url}{endpoint}"
            task.http_request.headers = {"Content-Type": "application/json"}
            task.http_request.body = json.dumps(payload).encode()

            # Set OIDC token for authentication
            task.http_request.oidc_token = tasks_v2.OidcToken()
            task.http_request.oidc_token.service_account_email = service_account_email

            # Add custom task ID if provided
            if task_id is None:
                task_id = generate_random_id()
            task.name = f"{queue_path}/tasks/{task_id}"

            # Add delay if specified
            if delay_seconds > 0:
                timestamp = timestamp_pb2.Timestamp()
                timestamp.FromSeconds(int(time.time()) + delay_seconds)
                task.schedule_time = timestamp

            # Create the task
            response = self.client.create_task(parent=queue_path, task=task)

            logger.info(
                f"create_task - Created task in {queue_type.value}: {response.name}"
            )

            # Create a firebase document for this
            self.poll_manager.create_task_record(
                task_id,
                queue_type,
                TaskStatus.PENDING,
                endpoint=endpoint,
                payload=json.dumps(payload),
            )
            logger.info(f"create_task - Created task record {task_id} in firebase")

            return response.name

        except Exception as e:
            logger.error(
                f"create_task - Failed to create task in {queue_type.value}: {e}"
            )
            raise

    def create_clip_inference_task(
        self,
        video_file: str,
        shot_idx: int,
        stringified_hoops: str,
        start_ts: float,
        end_ts: float,
        ball_key_ts: float,
        frame_sample_rate: int = 2,
        min_shot_score: float = 0.5,
        min_ball_score: float = 0.4,
        min_person_box_score: float = 0.7,
        min_person_keypoint_score: float = 0.1,
        min_segment_skeleton_iou: float = 0.6,
        dist_pixel_thres: int = 10,
        max_ball_person_ratio: int = 30,
        firebase_bucket: str = "",
    ):
        """Create clip inference task.

        Args:
            video_file: Raw video file
            shot_idx: Index for the shot
            stringified_hoops: List of detected hoop locations
            start_ts: Start timestamp to do inference for
            end_ts: End timestamp to do inference for
            ball_key_ts: Timestamp for ball keyframe
            frame_sample_rate: Sample rate for image frames
            min_shot_score: Minimum confidence to classify shot as a make
            min_ball_score: Minimum probability cutoff to consider a ball
            min_person_box_score: Minimum confidence to keep a predicted bounding box
            min_person_keypoint_score: Minimum confidence to keep a predicted keypoint
            min_segment_skeleton_iou: Minimum IoU to match a segment to a skeleton
            dist_pixel_thres: Max euclidean distance from ball to wrist to be a holder
            max_ball_person_ratio: Max percentage ratio for ball to person to be a holder
            firebase_bucket: Path to firebase bucket to upload artifacts

        Returns:
            task_name: Full name of created task
        """
        payload = {
            "video_file": video_file,
            "shot_idx": shot_idx,
            "stringified_hoops": stringified_hoops,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "ball_key_ts": ball_key_ts,
            "frame_sample_rate": frame_sample_rate,
            "min_shot_score": min_shot_score,
            "min_ball_score": min_ball_score,
            "min_person_box_score": min_person_box_score,
            "min_person_keypoint_score": min_person_keypoint_score,
            "min_segment_skeleton_iou": min_segment_skeleton_iou,
            "dist_pixel_thres": dist_pixel_thres,
            "max_ball_person_ratio": max_ball_person_ratio,
            "firebase_bucket": firebase_bucket,
        }
        task_id = generate_random_id()
        return self.create_task(
            queue_type=QueueType.CLIP_INFERENCE,
            endpoint="/infer/clip",
            payload=payload,
            task_id=task_id,
        )

    def create_fast_hoop_detection_task(
        self,
        video_file: str,
        vid_stride: int = 15,
        min_score: float = 0.3,
        stringified_annotated_points: str = "",
        stringified_override_boxes: str = "",
    ):
        """Create fast hoop detection task.

        Args:
            video_file: Raw video file
            vid_stride: Frame sample rate
            min_score: Minimum probability cutoff to consider a hoop
            stringified_annotated_points: Stringified annotations of hoop points
            stringified_override_boxes: Stringified overrides of hoop boxes

        Returns:
            task_name: Full name of created task
        """
        payload = {
            "video_file": video_file,
            "vid_stride": vid_stride,
            "min_score": min_score,
            "stringified_annotated_points": stringified_annotated_points,
            "stringified_override_boxes": stringified_override_boxes,
        }
        task_id = generate_random_id()
        return self.create_task(
            queue_type=QueueType.CLIP_INFERENCE,
            endpoint="/infer",
            payload=payload,
            task_id=task_id,
        )

    def create_slow_hoop_detection_task(
        self,
        video_file: str,
        vid_stride: int = 15,
        min_score: float = 0.3,
        stringified_annotated_points: str = "",
        stringified_override_boxes: str = "",
    ):
        """Create slow hoop detection task.

        Args:
            video_file: Raw video file
            vid_stride: Frame sample rate
            min_score: Minimum probability cutoff to consider a hoop
            stringified_annotated_points: Stringified annotations of hoop points
            stringified_override_boxes: Stringified overrides of hoop boxes

        Returns:
            task_name: Full name of created task
        """
        payload = {
            "video_file": video_file,
            "vid_stride": vid_stride,
            "min_score": min_score,
            "stringified_annotated_points": stringified_annotated_points,
            "stringified_override_boxes": stringified_override_boxes,
        }
        task_id = generate_random_id()
        return self.create_task(
            queue_type=QueueType.CLIP_INFERENCE,
            endpoint="/infer",
            payload=payload,
            task_id=task_id,
        )

    def create_video_processing_task(
        self,
        session_id: str,
        max_concurrent_tasks: int = 8,
        override_run_version: Optional[int] = None,
        inference_engine: str = "replicate",
    ) -> str:
        """Create video inference task.

        Args:
            session_id (str): Session ID.
            max_concurrent_tasks (int): Maximum number of concurrent Replicate tasks.
            override_run_version (Optional[int]): Override run version if provided.
            inference_engine (str): replicate | cloudrun

        Returns:
            str: Task name.
        """
        payload = {
            "session_id": session_id,
            "cache_dir": "./cache",
            "max_concurrent_tasks": max_concurrent_tasks,
            "override_run_version": override_run_version,
            "inference_engine": inference_engine,
        }
        task_id = generate_random_id()
        return self.create_task(
            queue_type=QueueType.VIDEO_PROCESSING,
            endpoint="/session/process",
            payload=payload,
            task_id=task_id,
        )

    def create_video_compression_task(
        self,
        session_id: str,
    ) -> str:
        """Create video compression task.

        Args:
            session_id (str): Session ID.

        Returns:
            str: Task name.
        """
        payload = {
            "session_id": session_id,
            "cache_dir": "./cache",
        }
        task_id = generate_random_id()
        return self.create_task(
            queue_type=QueueType.VIDEO_COMPRESSION,
            endpoint="/session/compress",
            payload=payload,
            task_id=task_id,
        )

    def create_combine_processing_task(
        self,
        video_id: str,
        combine_id: str,
        event_id: str,
    ) -> str:
        """Create combine processing task.

        Args:
            video_id (str): Video ID.
            combine_id (str): Combine ID.
            event_id (str): Event ID.

        Returns:
            str: Task name.
        """
        payload = {
            "video_id": video_id,
            "combine_id": combine_id,
            "event_id": event_id,
            "cache_dir": "./cache",
        }
        task_id = generate_random_id()
        return self.create_task(
            queue_type=QueueType.COMBINE,
            endpoint="/combine/process",
            payload=payload,
            task_id=task_id,
        )

    def create_combine_email_task(
        self,
        combine_id: str,
    ) -> str:
        """Create combine email task.

        Args:
            combine_id (str): Combine ID.

        Returns:
            str: Task name.
        """
        payload = {
            "combine_id": combine_id,
        }
        task_id = generate_random_id()
        return self.create_task(
            queue_type=QueueType.COMBINE,
            endpoint="/combine/email",
            payload=payload,
            task_id=task_id,
        )

    def create_message_username_task(
        self,
        user_id: str,
        delay_seconds: int = 86400,
    ) -> str:
        """Create message username task.

        Args:
            user_id (str): User ID.
            delay_seconds (int): Delay in seconds.

        Returns:
            str: Task name.
        """
        payload = {
            "user_id": user_id,
        }
        task_id = generate_random_id()
        return self.create_task(
            queue_type=QueueType.APP_NOTIFICATION,
            endpoint="/message/username",
            payload=payload,
            delay_seconds=delay_seconds,
            task_id=task_id,
        )

    def create_message_upload_task(
        self,
        user_id: str,
        session_id: str,
        delay_seconds: int = 86400,
    ) -> str:
        """Create message upload task.

        Args:
            user_id (str): User ID.
            session_id (str): Session ID.
            delay_seconds (int): Delay in seconds.

        Returns:
            str: Task name.
        """
        payload = {
            "user_id": user_id,
            "session_id": session_id,
        }
        task_id = generate_random_id()
        return self.create_task(
            queue_type=QueueType.APP_NOTIFICATION,
            endpoint="/message/upload",
            payload=payload,
            delay_seconds=delay_seconds,
            task_id=task_id,
        )


def generate_random_id() -> str:
    """Generate ID using Firebase's 62-character alphabet"""
    # Same alphabet Firebase uses for auto-generated IDs
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(secrets.choice(alphabet) for _ in range(20))
