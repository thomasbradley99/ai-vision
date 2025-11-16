import asyncio
import replicate
from os import environ
from os.path import join
from httpx import Timeout
from typing import List, Dict, Optional, Callable, Any
from replicate.client import Client as ReplicateClient
from replicate.exceptions import ModelError as ReplicateModelError
from shared_utils.constants import (
    GCLOUD_DEFAULT_LOCATION,
    GCLOUD_COMPUTE_PROJECT_ID,
)
from shared_utils import (
    get_logger,
    StructuredLogger,
    CloudRunManager,
    CloudTasksConfig,
)
from shared_utils.utils import tick
from shared_utils.types import (
    ClipOutput,
    RawClipOutput,
    WrappedClipOutput,
    HoopOutput,
    RawHoopOutput,
    WrappedHoopOutput,
)
from .utils import list_to_comma_sep_str, stringify_hoops

# Initialize loggers
logger = get_logger(__name__)
struct_logger = StructuredLogger(__name__)


# --- hoop detection ---


async def replicate_hoop_detection_in_background(
    client: ReplicateClient,
    model_name: str,
    model_version: str,
    input_dict: Dict,
    poll_interval: int = 10,
) -> WrappedHoopOutput:
    """Wrapper around replicate to call a deployed model to do inference.

    Args:
        client: Replicate client
        model_name: Replicate model name
        model_version: Replicate model version
        input_dict: Replicate input dictionary

    Return:
        WrappedHoopOutput

    Note:
        Requires the REPLICATE_API_TOKEN global var to be set
    """
    logger.info(
        f"replicate_hoop_detection_in_background - API token: {environ['REPLICATE_API_TOKEN']}",
    )
    start_time = tick()
    # Initialize default values
    hoops = None
    frac_missing = None
    frac_moving = None
    success = False
    error = ""
    try:
        model = replicate.models.get(model_name)
        version = model.versions.get(model_version)
        prediction = client.predictions.create(version=version, input=input_dict)
        # Use async_wait to asynchronously wait for the prediction to complete
        # This will poll until the job is done
        start_poll_time = tick()
        while prediction.status not in ["succeeded", "failed", "canceled"]:
            await asyncio.sleep(poll_interval)
            prediction.reload()  # Refresh the prediction status
            elapsed_poll_time = tick() - start_poll_time
            logger.info(
                f"replicate_hoop_detection_in_background - prediction {prediction.id} | "
                f"status: {prediction.status} | "
                f"elapsed: {elapsed_poll_time:.1f}s | "
                f"still running..."
            )
        logger.info(
            f"replicate_hoop_detection_in_background - prediction {prediction.id} | "
            f"status: {prediction.status} | "
            f"started at: {prediction.started_at} | "
            f"created at: {prediction.created_at} | "
            f"completed at: {prediction.completed_at}"
        )
        if prediction.status == "succeeded":
            raw_output = RawHoopOutput.model_validate(prediction.output)
            raw_hoops = raw_output.hoops
            frac_missing = raw_output.frac_missing
            frac_moving = raw_output.frac_moving
            if not raw_hoops or len(raw_hoops) == 0:
                error = "No hoops found"
                logger.error(f"replicate_hoop_detection_in_background - {error}")
            else:
                success = True
                hoops = [HoopOutput.model_validate(fpr) for fpr in raw_hoops]
        elif prediction.status == "failed":
            error = f"prediction {prediction.id} failed with error: {prediction.error}"
            logger.error(f"replicate_hoop_detection_in_background - {error}")
        elif prediction.status == "canceled":
            error = f"prediction {prediction.id} was canceled"
            logger.error(f"replicate_hoop_detection_in_background - {error}")
        else:
            error = f"prediction {prediction.id} experienced an unknown error"
            logger.error(f"replicate_hoop_detection_in_background - {error}")
    except ReplicateModelError as e:
        error = str(e)
        logger.error(f"replicate_hoop_detection_in_background - {error}")

    end_time = tick()
    elapsed_time = end_time - start_time
    output = {
        "hoops": hoops,
        "frac_missing": frac_missing,
        "frac_moving": frac_moving,
        "success": success,
        "error": error,
        "elapsed_sec": elapsed_time,
    }
    return WrappedHoopOutput.model_validate(output)


async def async_hoop_detection_with_replicate(
    replicate_api_token: str,
    replicate_hoop_model: str,
    video_url: str,
    vid_stride: int = 10,
    min_score: float = 0.3,
    annotated_points: Optional[List[float] | List[int]] = None,
    override_boxes: Optional[List[float] | List[int]] = None,
    replicate_timeout: int = 10,
    poll_interval: int = 10,
) -> WrappedHoopOutput:
    """Submit a hoop detection to Replicate and wait for results.
    This function will hang until completion.

    Args:
        replicate_api_token: API Token for Replicate
        replicate_hoop_model: Model identifier for hoop model on Replicate
        replicate_timeout: Minutes to wait
        poll_interval: Seconds in between each poll

    Return:
        WrappedHoopOutput
    """
    # Set the environment var
    environ['REPLICATE_API_TOKEN'] = replicate_api_token
    # Build input dict
    replicate_input = {
        "video_file": video_url,
        "vid_stride": vid_stride,
        "min_score": min_score,
        "stringified_annotated_points": list_to_comma_sep_str(annotated_points)
        if annotated_points is not None
        else "",
        "stringified_override_boxes": list_to_comma_sep_str(override_boxes)
        if override_boxes is not None
        else "",
    }
    struct_logger.info(
        "async_hoop_detection_with_replicate - Building Replicate input",
        **replicate_input,
    )
    # Build client with timeout config
    timeout = Timeout(
        connect=replicate_timeout * 60,
        read=replicate_timeout * 3 * 60,
        write=100,
        pool=10,
    )
    client = ReplicateClient(api_token=replicate_api_token, timeout=timeout)
    model_name, model_version = replicate_hoop_model.split(":")
    struct_logger.info(
        "async_hoop_detection_with_replicate - Building Replicate client",
        replicate_api_token=replicate_api_token,
        model_name=model_name,
        model_version=model_version,
    )
    # Run in background
    result = await replicate_hoop_detection_in_background(
        client, model_name, model_version, replicate_input,
        poll_interval=poll_interval,
    )
    return result


async def async_hoop_detection_with_cloudrun(
    cloudrun_service_url: str,
    cloudrun_invoker_sa: str,
    video_url: str,
    vid_stride: int = 10,
    min_score: float = 0.3,
    annotated_points: Optional[List[float] | List[int]] = None,
    override_boxes: Optional[List[float] | List[int]] = None,
    cloudrun_timeout: int = 10,
    poll_interval: int = 10,
    use_fast: bool = True,
) -> WrappedHoopOutput:
    """Submit a hoop detection to a Cloud Run service and wait for results.
    This function will hang until completion.

    Args:
        cloudrun_service_url: URL for the deployed service
        cloudrun_invoker_sa: Service account email
        cloudrun_timeout: Minutes while polling before timeout
        poll_interval: Seconds in between each poll
        use_fast: Whether to use the fast or slow model.

    Return:
        WrappedHoopOutput
    """
    stringified_annotated_points = (
        list_to_comma_sep_str(annotated_points)
        if annotated_points is not None
        else ""
    )
    stringified_override_boxes = (
        list_to_comma_sep_str(override_boxes)
        if override_boxes is not None
        else ""
    )

    config = CloudTasksConfig(
        project_id=GCLOUD_COMPUTE_PROJECT_ID,
        location=GCLOUD_DEFAULT_LOCATION,
        cloudrun_gpu_service_url=cloudrun_service_url,
        cloudrun_gpu_invoker_sa=cloudrun_invoker_sa,
    )
    client = CloudRunManager(config=config)

    if use_fast:
        output = await client.trigger_fast_hoop_detection_and_wait(
            video_url,
            vid_stride=vid_stride,
            min_score=min_score,
            stringified_annotated_points=stringified_annotated_points,
            stringified_override_boxes=stringified_override_boxes,
            timeout_minutes=cloudrun_timeout,
            poll_interval_seconds=poll_interval,
        )
    else:
        output = await client.trigger_slow_hoop_detection_and_wait(
            video_url,
            vid_stride=vid_stride,
            min_score=min_score,
            stringified_annotated_points=stringified_annotated_points,
            stringified_override_boxes=stringified_override_boxes,
            timeout_minutes=cloudrun_timeout,
            poll_interval_seconds=poll_interval,
        )

    return output


# --- clip inference ---


def create_clip_inference_input(
    video_url: str,
    video_length: float,
    shot_idx: int,
    hoops: List[HoopOutput],
    ball_key_ts: float,
    firebase_bucket: Optional[str] = None,
    frame_sample_rate: int = 2,
    context_window: int = 4,
    min_shot_score: float = 0.5,
    min_ball_score: float = 0.2,
    min_person_box_score: float = 0.1,
    min_person_keypoint_score: float = 0.1,
    min_segment_skeleton_iou: float = 0.6,
    dist_pixel_thres: int = 10,
    max_ball_person_ratio: int = 30,
) -> Dict[str, Any]:
    """Build a replicate input to be executed."""
    start_ts = round(max(0, ball_key_ts - context_window), 3)
    end_ts = round(min(video_length, ball_key_ts + context_window), 3)
    clip_hoops = [hoop for hoop in hoops if start_ts <= hoop.ts <= end_ts]
    stringified_clip_hoops = stringify_hoops(clip_hoops)

    if firebase_bucket is not None:
        shot_bucket = join(firebase_bucket, f"shot-{shot_idx}")
    else:
        shot_bucket = None

    replicate_input = {
        "video_file": video_url,
        "shot_idx": shot_idx,
        "stringified_hoops": stringified_clip_hoops,
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
        "firebase_bucket": shot_bucket if shot_bucket is not None else "",
    }
    return replicate_input



async def replicate_clip_inference_in_background(
    client: ReplicateClient,
    model_name: str,
    model_version: str,
    input_dict: Dict,
    poll_interval: int = 10,
) -> WrappedClipOutput:
    """Wrapper around replicate to run in the background.

    Args:
        client: Replicate client
        model_name: Replicate model name
        model_version: Replicate model version
        input_dict: Replicate input dictionary
        poll_interval: Seconds in between each poll

    Return:
        WrappedClipOutput

    Note:
        Requires the REPLICATE_API_TOKEN global var to be set
    """
    start_time = tick()
    shot_idx = int(input_dict["shot_idx"])
    logger.info(
        f"[shot {shot_idx}] replicate_clip_inference_in_background - API token: {environ['REPLICATE_API_TOKEN']}",
    )
    try:
        model = replicate.models.get(model_name)
        version = model.versions.get(model_version)
        prediction = client.predictions.create(version=version, input=input_dict)
        # Use async_wait to asynchronously wait for the prediction to complete
        # This will poll until the job is done
        start_poll_time = tick()
        while prediction.status not in ["succeeded", "failed", "canceled"]:
            await asyncio.sleep(poll_interval)
            prediction.reload()  # Refresh the prediction status
            elapsed_poll_time = tick() - start_poll_time
            logger.info(
                f"[shot {shot_idx}] replicate_clip_inference_in_background - prediction {prediction.id} | "
                f"status: {prediction.status} | "
                f"elapsed: {elapsed_poll_time:.1f}s | "
                f"still running..."
            )
        logger.info(
            f"[shot {shot_idx}] replicate_clip_inference_in_background - prediction {prediction.id} | "
            f"status: {prediction.status} | "
            f"created at: {prediction.created_at} | "
            f"completed at: {prediction.completed_at}"
        )
        if prediction.status == "succeeded":
            raw_output = RawClipOutput.model_validate(prediction.output)
            if not raw_output:
                # This is unexpected and should never really happen. Maybe happens if Replicate breaks.
                success = False
                error = "Output is unexpectedly empty"
                output = None
                logger.error(f"replicate_clip_inference_in_background - {error}")
            elif raw_output.output is None:
                # This happens when the shot fails to process due to being a false positive (common).
                success = False
                error = "Output is empty. System decided shot to be a false positive."
                output = None
                logger.error(f"replicate_clip_inference_in_background - {error}")
            else:
                # This happens when the shot succeeds.
                success = True
                error = ""
                output = ClipOutput.model_validate(raw_output.output)
        elif prediction.status == "failed":
            success = False
            error = f"Prediction failed with error: {prediction.error}"
            output = None
            logger.error(f"[shot {shot_idx}] replicate_clip_inference_in_background - {error}")
        elif prediction.status == "canceled":
            success = False
            error = "Prediction was canceled"
            output = None
            logger.error(f"[shot {shot_idx}] replicate_clip_inference_in_background - {error}")
        else:
            success = False
            error = "Unknown error occurred"
            output = None
            logger.error(f"[shot {shot_idx}] replicate_clip_inference_in_background - {error}")
    except ReplicateModelError as e:
        success = False
        error = str(e)
        output = None
        logger.error(f"[shot {shot_idx}] replicate_clip_inference_in_background - {error}")
    end_time = tick()
    elapsed_time = end_time - start_time
    run_output = {
        "shot_idx": shot_idx,
        "output": output,
        "success": success,
        "error": error,
        "elapsed_sec": elapsed_time,
    }
    return WrappedClipOutput.model_validate(run_output)


async def async_many_clip_inference_with_replicate(
    replicate_api_token: str,
    replicate_clip_model: str,
    video_url: str,
    video_length: float,
    hoops: List[HoopOutput],
    ball_key_ts_above: List[float],
    firebase_bucket: Optional[str] = None,
    frame_sample_rate: int = 2,
    context_window: int = 4,
    min_shot_score: float = 0.5,
    min_ball_score: float = 0.2,
    min_person_box_score: float = 0.1,
    min_person_keypoint_score: float = 0.1,
    min_segment_skeleton_iou: float = 0.6,
    dist_pixel_thres: int = 10,
    max_ball_person_ratio: int = 30,
    replicate_timeout: int = 10,
    poll_interval: int = 10,
    max_concurrent_tasks: int = 8,
    task_callback: Optional[Callable[[asyncio.Task[WrappedClipOutput]], None]] = None,
) -> List[WrappedClipOutput]:
    """Do inference on all the chunks asynchronously with a limit on concurrent jobs.

    Args:
        replicate_api_token:
        replicate_clip_model:
        task_callback: Optional callback function to call after each task is done

    Return:
        List of WrappedClipOutput
    """
    # Set the environment var
    environ['REPLICATE_API_TOKEN'] = replicate_api_token

    model_name, model_version = replicate_clip_model.split(":")
    logger.info(f"async_many_clip_inference_with_replicate - using model: {model_name}:{model_version}")

    # Semaphore to limit the number of concurrent jobs
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def sem_task(model_name: str, model_version: str, input_dict: Dict):
        """Semaphore to limit the number of concurrent jobs."""
        # Build client with timeout config
        # NOTE: important that this is done PER sem task
        timeout = Timeout(
            connect=replicate_timeout * 60,
            read=replicate_timeout * 3 * 60,
            write=100,
            pool=10,
        )
        client = ReplicateClient(api_token=replicate_api_token, timeout=timeout)
        async with semaphore:  # Each task must acquire the semaphore before running
            return await replicate_clip_inference_in_background(
                client, model_name, model_version, input_dict,
                poll_interval=poll_interval,
            )

    tasks = []  # Create tasks
    for shot_idx, ball_key_ts in enumerate(ball_key_ts_above):
        input_dict = create_clip_inference_input(
            video_url,
            video_length,
            shot_idx,
            hoops,
            ball_key_ts,
            frame_sample_rate=frame_sample_rate,
            firebase_bucket=firebase_bucket,
            context_window=context_window,
            min_shot_score=min_shot_score,
            min_ball_score=min_ball_score,
            min_person_box_score=min_person_box_score,
            min_person_keypoint_score=min_person_keypoint_score,
            min_segment_skeleton_iou=min_segment_skeleton_iou,
            dist_pixel_thres=dist_pixel_thres,
            max_ball_person_ratio=max_ball_person_ratio,
        )
        struct_logger.info(
            "async_many_clip_inference_with_replicate - creating clip inference async task",
            shot_idx=shot_idx,
            ball_key_ts=ball_key_ts,
        )
        # Use sem_task to limit the number of concurrent jobs
        task = asyncio.create_task(sem_task(model_name, model_version, input_dict))
        if task_callback:
            task.add_done_callback(task_callback)
        tasks.append(task)

    # Gather results from all tasks
    results = await asyncio.gather(*tasks)
    return results


async def async_many_clip_inference_with_cloudrun(
    cloudrun_service_url: str,
    cloudrun_invoker_sa: str,
    video_url: str,
    video_length: float,
    hoops: List[HoopOutput],
    ball_key_ts_above: List[float],
    firebase_bucket: Optional[str] = None,
    frame_sample_rate: int = 2,
    context_window: int = 4,
    min_shot_score: float = 0.5,
    min_ball_score: float = 0.2,
    min_person_box_score: float = 0.1,
    min_person_keypoint_score: float = 0.1,
    min_segment_skeleton_iou: float = 0.6,
    dist_pixel_thres: int = 10,
    max_ball_person_ratio: int = 30,
    cloudrun_timeout: int = 10,
    max_concurrent_tasks: int = 8,
    task_callback: Optional[Callable[[asyncio.Task[WrappedClipOutput]], None]] = None,
) -> List[WrappedClipOutput]:
    """Do inference on all the chunks asynchronously with a limit on concurrent jobs.

    Args:
        cloudrun_service_url:
        cloudrun_invoker_sa:
        task_callback: Optional callback function to call after each task is done

    Return:
        List of WrappedClipOutput
    """
    # Semaphore to limit the number of concurrent jobs
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def sem_task(input_dict: Dict):
        """Semaphore to limit the number of jobs"""

        config = CloudTasksConfig(
            project_id=GCLOUD_COMPUTE_PROJECT_ID,
            location=GCLOUD_DEFAULT_LOCATION,
            cloudrun_gpu_service_url=cloudrun_service_url,
            cloudrun_gpu_invoker_sa=cloudrun_invoker_sa,
        )
        client = CloudRunManager(config=config)
        async with semaphore:  # Each task must acquire the semaphore before running
            return await client.trigger_clip_inference_and_wait(
                **input_dict,
                timeout_minutes=cloudrun_timeout,
                poll_interval_seconds=3,
            )

    tasks = []  # Create tasks
    for shot_idx, ball_key_ts in enumerate(ball_key_ts_above):
        input_dict = create_clip_inference_input(
            video_url,
            video_length,
            shot_idx,
            hoops,
            ball_key_ts,
            frame_sample_rate=frame_sample_rate,
            firebase_bucket=firebase_bucket,
            context_window=context_window,
            min_shot_score=min_shot_score,
            min_ball_score=min_ball_score,
            min_person_box_score=min_person_box_score,
            min_person_keypoint_score=min_person_keypoint_score,
            min_segment_skeleton_iou=min_segment_skeleton_iou,
            dist_pixel_thres=dist_pixel_thres,
            max_ball_person_ratio=max_ball_person_ratio,
        )
        # Use sem_task to limit the number of concurrent jobs
        task = asyncio.create_task(sem_task(input_dict))
        if task_callback:
            task.add_done_callback(task_callback)
        tasks.append(task)

    # Gather results from all tasks
    results = await asyncio.gather(*tasks)
    return results
