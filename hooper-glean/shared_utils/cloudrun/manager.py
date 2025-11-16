import json
import asyncio
import requests
from typing import Optional, Dict, Any
from ..cloudtasks.manager import CloudTasksManager, CloudTasksConfig
from ..logger import get_logger
from ..utils import get_task_id_from_name, tick
from ..types import WrappedHoopOutput, WrappedClipOutput, RawHoopOutput, RawClipOutput
from ..poll import TaskStatus

logger = get_logger(__name__)


class CloudRunManager:
    """Client for triggering Cloud Run jobs via Cloud Tasks."""

    def __init__(self, config: CloudTasksConfig):
        self.task_manager = CloudTasksManager(config)
        self.config = config

    def trigger_clip_inference(
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
    ) -> str:
        """Trigger clip inference on Cloud Run (GPU)"""
        try:
            task_name = self.task_manager.create_clip_inference_task(
                video_file=video_file,
                shot_idx=shot_idx,
                stringified_hoops=stringified_hoops,
                start_ts=start_ts,
                end_ts=end_ts,
                ball_key_ts=ball_key_ts,
                frame_sample_rate=frame_sample_rate,
                min_shot_score=min_shot_score,
                min_ball_score=min_ball_score,
                min_person_box_score=min_person_box_score,
                min_person_keypoint_score=min_person_keypoint_score,
                min_segment_skeleton_iou=min_segment_skeleton_iou,
                dist_pixel_thres=dist_pixel_thres,
                max_ball_person_ratio=max_ball_person_ratio,
                firebase_bucket=firebase_bucket,
            )
            logger.info(
                f"trigger_clip_inference - Triggered clip inference: {task_name}"
            )
            task_id = get_task_id_from_name(task_name)
            return task_id
        except Exception as e:
            logger.error(
                f"trigger_clip_inference - Failed to trigger clip inference: {e}"
            )
            raise

    def trigger_fast_hoop_detection(
        self,
        video_file: str,
        vid_stride: int = 15,
        min_score: float = 0.3,
        stringified_annotated_points: str = "",
        stringified_override_boxes: str = "",
    ) -> str:
        """Trigger faster but frequent hoop detection on Cloud Run (GPU)"""
        try:
            task_name = self.task_manager.create_fast_hoop_detection_task(
                video_file=video_file,
                vid_stride=vid_stride,
                min_score=min_score,
                stringified_annotated_points=stringified_annotated_points,
                stringified_override_boxes=stringified_override_boxes,
            )
            logger.info(
                f"trigger_fast_hoop_detection - Triggered hoop detection: {task_name}"
            )
            task_id = get_task_id_from_name(task_name)
            return task_id
        except Exception as e:
            logger.error(
                f"trigger_fast_hoop_detection - Failed to trigger hoop detection: {e}"
            )
            raise

    def trigger_slow_hoop_detection(
        self,
        video_file: str,
        vid_stride: int = 15,
        min_score: float = 0.3,
        stringified_annotated_points: str = "",
        stringified_override_boxes: str = "",
    ) -> str:
        """Trigger slow but infrequent hoop detection on Cloud Run (GPU)"""
        try:
            task_name = self.task_manager.create_slow_hoop_detection_task(
                video_file=video_file,
                vid_stride=vid_stride,
                min_score=min_score,
                stringified_annotated_points=stringified_annotated_points,
                stringified_override_boxes=stringified_override_boxes,
            )
            logger.info(
                f"trigger_slow_hoop_detection - Triggered hoop detection: {task_name}"
            )
            task_id = get_task_id_from_name(task_name)
            return task_id
        except Exception as e:
            logger.error(
                f"trigger_slow_hoop_detection - Failed to trigger hoop detection: {e}"
            )
            raise

    def trigger_video_processing(
        self,
        session_id: str,
        max_concurrent_tasks: int = 8,
        override_run_version: Optional[int] = None,
        inference_engine: str = "replicate",
    ) -> str:
        """Trigger video processing on Cloud Run"""

        if inference_engine not in ["replicate", "cloudrun"]:
            logger.warning(
                f"trigger_video_processing - Inference engine {inference_engine} not supported. Defaulting to replicate."
            )
            inference_engine = "replicate"

        try:
            task_name = self.task_manager.create_video_processing_task(
                session_id=session_id,
                max_concurrent_tasks=max_concurrent_tasks,
                override_run_version=override_run_version,
                inference_engine=inference_engine,
            )
            logger.info(
                f"trigger_video_processing - Triggered video processing for session {session_id}: {task_name}"
            )
            task_id = get_task_id_from_name(task_name)
            return task_id
        except Exception as e:
            logger.error(
                f"trigger_video_processing - Failed to trigger video processing for session {session_id}: {e}"
            )
            raise

    def trigger_video_compression(self, session_id: str) -> str:
        """Trigger video compression on Cloud Run"""
        try:
            task_name = self.task_manager.create_video_compression_task(
                session_id=session_id
            )
            logger.info(
                f"trigger_video_compression - Triggered video compression for session {session_id}: {task_name}"
            )
            task_id = get_task_id_from_name(task_name)
            return task_id
        except Exception as e:
            logger.error(
                f"trigger_video_compression - Failed to trigger video compression for session {session_id}: {e}"
            )
            raise

    def trigger_combine_processing(
        self, video_id: str, combine_id: str, event_id: str
    ) -> str:
        """Trigger combine processing on Cloud Run"""
        try:
            task_name = self.task_manager.create_combine_processing_task(
                video_id=video_id, combine_id=combine_id, event_id=event_id
            )
            logger.info(
                f"trigger_combine_processing - Triggered combine processing: {task_name}"
            )
            task_id = get_task_id_from_name(task_name)
            return task_id
        except Exception as e:
            logger.error(
                f"trigger_combine_processing - Failed to trigger combine processing: {e}"
            )
            raise

    def trigger_combine_email(self, combine_id: str) -> str:
        """Trigger combine email on Cloud Run"""
        try:
            task_name = self.task_manager.create_combine_email_task(
                combine_id=combine_id
            )
            logger.info(f"trigger_combine_email - Triggered combine email: {task_name}")
            task_id = get_task_id_from_name(task_name)
            return task_id
        except Exception as e:
            logger.error(
                f"trigger_combine_email - Failed to trigger combine email: {e}"
            )
            raise

    def trigger_message_username(self, user_id: str, delay_seconds: int = 86400) -> str:
        """Trigger message username on Cloud Run"""
        try:
            task_name = self.task_manager.create_message_username_task(
                user_id=user_id,
                delay_seconds=delay_seconds,
            )
            logger.info(
                f"trigger_message_username - Triggered message username: {task_name}"
            )
            task_id = get_task_id_from_name(task_name)
            return task_id
        except Exception as e:
            logger.error(
                f"trigger_message_username - Failed to trigger message username: {e}"
            )
            raise

    def trigger_message_upload(
        self, user_id: str, session_id: str, delay_seconds: int = 86400
    ) -> str:
        """Trigger message upload on Cloud Run"""
        try:
            task_name = self.task_manager.create_message_upload_task(
                user_id=user_id,
                session_id=session_id,
                delay_seconds=delay_seconds,
            )
            logger.info(
                f"trigger_message_upload - Triggered message upload: {task_name}"
            )
            task_id = get_task_id_from_name(task_name)
            return task_id
        except Exception as e:
            logger.error(
                f"trigger_message_upload - Failed to trigger message upload: {e}"
            )
            raise

    async def trigger_clip_inference_and_wait(
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
        timeout_minutes: int = 10,
        poll_interval_seconds: int = 3,
    ) -> WrappedClipOutput:
        """Trigger clip inference on Cloud Run and poll until the result is complete or timeout is reached."""

        start_time = tick()
        try:
            task_name = self.task_manager.create_clip_inference_task(
                video_file=video_file,
                shot_idx=shot_idx,
                stringified_hoops=stringified_hoops,
                start_ts=start_ts,
                end_ts=end_ts,
                ball_key_ts=ball_key_ts,
                frame_sample_rate=frame_sample_rate,
                min_shot_score=min_shot_score,
                min_ball_score=min_ball_score,
                min_person_box_score=min_person_box_score,
                min_person_keypoint_score=min_person_keypoint_score,
                min_segment_skeleton_iou=min_segment_skeleton_iou,
                dist_pixel_thres=dist_pixel_thres,
                max_ball_person_ratio=max_ball_person_ratio,
                firebase_bucket=firebase_bucket,
            )
            logger.info(
                f"trigger_clip_inference_and_wait - Triggered clip inference: {task_name}"
            )
        except Exception as e:
            error_msg = f"trigger_clip_inference_and_wait - Failed to trigger clip inference: {e}"
            logger.error(error_msg)
            end_time = tick()
            return WrappedClipOutput(
                shot_idx=shot_idx,
                success=False,
                error=error_msg,
                elapsed_sec=end_time - start_time,
            )

        # Poll for completion using task_name
        raw_output_dict = await self._poll_for_completion(
            task_name, timeout_minutes, poll_interval_seconds
        )
        logger.info("trigger_clip_inference_and_wait - Poll completed")

        try:  # Attempt to cast to type
            raw_output = RawClipOutput(**raw_output_dict)
        except TypeError as e:
            logger.error(
                f"trigger_clip_inference_and_wait - Failed to cast result to RawClipOutput: {e}"
            )
            logger.error(f"Raw result keys: {list(raw_output_dict.keys())}")
            end_time = tick()
            return WrappedClipOutput(
                shot_idx=shot_idx,
                success=False,
                error=f"trigger_clip_inference_and_wait - Invalid result format: {e}",
                elapsed_sec=end_time - start_time,
            )

        end_time = tick()
        return WrappedClipOutput(
            shot_idx=raw_output.shot_idx,
            output=raw_output.output,
            elapsed_sec=end_time - start_time,
        )

    async def trigger_fast_hoop_detection_and_wait(
        self,
        video_file: str,
        vid_stride: int = 15,
        min_score: float = 0.3,
        stringified_annotated_points: str = "",
        stringified_override_boxes: str = "",
        timeout_minutes: int = 10,
        poll_interval_seconds: int = 3,
    ) -> WrappedHoopOutput:
        """Trigger fast hoop detection on Cloud Run and poll until the result is complete or timeout is reached."""

        start_time = tick()
        try:
            task_name = self.task_manager.create_fast_hoop_detection_task(
                video_file=video_file,
                vid_stride=vid_stride,
                min_score=min_score,
                stringified_annotated_points=stringified_annotated_points,
                stringified_override_boxes=stringified_override_boxes,
            )
            logger.info(
                f"trigger_fast_hoop_detection_and_wait - Triggered fast hoop detection task: {task_name}"
            )
        except Exception as e:
            error_msg = f"trigger_fast_hoop_detection_and_wait - Failed to trigger hoop detection: {e}"
            logger.error(error_msg)
            end_time = tick()
            return WrappedHoopOutput(
                success=False,
                error=error_msg,
                elapsed_sec=end_time - start_time,
            )

        # Poll for completion using task_name
        raw_output_dict = await self._poll_for_completion(
            task_name, timeout_minutes, poll_interval_seconds
        )
        logger.info("trigger_fast_hoop_detection_and_wait - Poll completed")

        try:  # Attempt to cast to type
            output = RawHoopOutput(**raw_output_dict)
        except TypeError as e:
            logger.error(
                f"trigger_fast_hoop_detection_and_wait - Failed to cast result to RawHoopOutput: {e}"
            )
            logger.error(f"Raw result keys: {list(raw_output_dict.keys())}")
            end_time = tick()
            return WrappedHoopOutput(
                success=False,
                error=f"trigger_fast_hoop_detection_and_wait - Invalid result format: {e}",
                elapsed_sec=end_time - start_time,
            )

        end_time = tick()
        return WrappedHoopOutput(
            hoops=output.hoops,
            frac_missing=output.frac_missing,
            frac_moving=output.frac_moving,
            elapsed_sec=end_time - start_time,
        )

    async def trigger_slow_hoop_detection_and_wait(
        self,
        video_file: str,
        vid_stride: int = 15,
        min_score: float = 0.3,
        stringified_annotated_points: str = "",
        stringified_override_boxes: str = "",
        timeout_minutes: int = 10,
        poll_interval_seconds: int = 3,
    ) -> WrappedHoopOutput:
        """Trigger slow hoop detection on Cloud Run and poll until the result is complete or timeout is reached."""

        start_time = tick()
        try:
            task_name = self.task_manager.create_slow_hoop_detection_task(
                video_file=video_file,
                vid_stride=vid_stride,
                min_score=min_score,
                stringified_annotated_points=stringified_annotated_points,
                stringified_override_boxes=stringified_override_boxes,
            )
            logger.info(
                f"trigger_slow_hoop_detection_and_wait - Triggered slow hoop detection task: {task_name}"
            )

            # Parse the task id from the task name
            task_id = get_task_id_from_name(task_name)
            logger.info(
                f"trigger_slow_hoop_detection_and_wait - Setting task {task_id} to PENDING"
            )
        except Exception as e:
            error_msg = f"trigger_slow_hoop_detection_and_wait - Failed to trigger hoop detection: {e}"
            logger.error(error_msg)
            end_time = tick()
            return WrappedHoopOutput(
                success=False,
                error=error_msg,
                elapsed_sec=end_time - start_time,
            )

        # Poll for completion
        raw_output = await self._poll_for_completion(
            task_id, timeout_minutes, poll_interval_seconds
        )
        logger.info("trigger_slow_hoop_detection_and_wait - Poll completed")

        try:  # Attempt to cast to type
            output = RawHoopOutput(**raw_output)
        except TypeError as e:
            logger.error(
                f"trigger_slow_hoop_detection_and_wait - Failed to cast result to RawHoopOutput: {e}"
            )
            logger.error(f"Raw result keys: {list(raw_output.keys())}")
            end_time = tick()
            return WrappedHoopOutput(
                success=False,
                error=f"trigger_slow_hoop_detection_and_wait - Invalid result format: {e}",
                elapsed_sec=end_time - start_time,
            )

        end_time = tick()
        return WrappedHoopOutput(
            hoops=output.hoops,
            frac_missing=output.frac_missing,
            frac_moving=output.frac_moving,
            elapsed_sec=end_time - start_time,
        )

    async def _poll_for_completion(
        self,
        task_id: str,
        timeout_minutes: int,
        poll_interval_seconds: int
    ) -> Dict:
        """Poll for task completion

        Args:
            task_id: Key used to fetch item from poll database
            timeout_minutes: number of minutes to wait before quitting
            poll_interval_seconds: re-ping every N seconds

        Returns:
            raw results
        """
        start_time = asyncio.get_event_loop().time()
        timeout_seconds = timeout_minutes * 60
        last_progress = 0.0

        while True:
            # Check if timeout exceeded
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_seconds:
                self.task_manager.poll_manager.update_task_record(
                    task_id,
                    TaskStatus.FAILED,
                    error="Timeout exceeded",
                )
                logger.error(
                    f"_poll_for_completion - Task {task_id} timed out after {timeout_minutes} minutes"
                )
                raise TimeoutError(
                    f"Task {task_id} timed out after {timeout_minutes} minutes"
                )

            try:
                # Check task status
                task_record = self.task_manager.get_task_record(task_id)

                if task_record is None:
                    raise ValueError("Document not found in firebase")

                if task_record.status == TaskStatus.COMPLETED:
                    if task_record.resultUrl is None:
                        raise Exception("Document is missing the result url")

                    logger.info(
                        f"_poll_for_completion - Task {task_id} completed successfully"
                    )
                    return self._parse_result(task_record.resultUrl)

                elif task_record.status == TaskStatus.FAILED:
                    raise RuntimeError(f"Task failed: {task_record.error}")

            except ValueError:
                # Task not found yet, continue polling
                if elapsed < 30:  # Give it 30 seconds before warning
                    pass
                else:
                    logger.warning(
                        f"_poll_for_completion - Task {task_id} not found in result store after {elapsed:.0f}s"
                    )

            except Exception as e:
                logger.error(
                    f"_poll_for_completion - Task {task_id} faced unknown error: {e}"
                )

            # Wait before next poll
            await asyncio.sleep(poll_interval_seconds)

    def _parse_result(self, result_url: str) -> Dict[str, Any]:
        """Download the data from the result URL and return as dict.

        Args:
            result_url: Firebase Storage public URL

        Returns:
            Parsed result dictionary

        Raises:
            requests.RequestException: If download fails
            json.JSONDecodeError: If JSON parsing fails
            ValueError: If URL is invalid or response is not JSON
        """
        try:
            # Download the file from Firebase Storage
            response = requests.get(result_url, timeout=30)
            response.raise_for_status()  # Raises HTTPError for bad responses

            # Parse JSON response
            result_data = response.json()

            logger.info(
                f"parse_result - Successfully downloaded result from {result_url}"
            )
            return result_data

        except requests.exceptions.Timeout:
            logger.error(f"parse_result - Timeout downloading from {result_url}")
            raise ValueError(f"Timeout downloading result from {result_url}")

        except requests.exceptions.HTTPError as e:
            logger.error(
                f"parse_result - HTTP error downloading from {result_url}: {e}"
            )
            raise ValueError(
                f"Failed to download result: HTTP {e.response.status_code}"
            )

        except requests.exceptions.RequestException as e:
            logger.error(
                f"parse_result - Request error downloading from {result_url}: {e}"
            )
            raise ValueError(f"Network error downloading result: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"parse_result - JSON decode error for {result_url}: {e}")
            raise ValueError(f"Invalid JSON in result file: {e}")

        except Exception as e:
            logger.error(
                f"parse_result - Unexpected error downloading from {result_url}: {e}"
            )
            raise ValueError(f"Unexpected error downloading result: {e}")
