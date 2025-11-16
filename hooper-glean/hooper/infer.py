import gc
import cv2
import torch
import shutil
import logging
import asyncio
import numpy as np
from tqdm import tqdm
from PIL import Image
from os import makedirs, remove
from os.path import join, isfile, splitext
import matplotlib.pyplot as plt
from torchvision.io import read_video
from typing import List, Dict, Tuple, Any, Optional, Union

from shared_utils.utils import get_firebase_client
from shared_utils.types import (
    PlayerEmbedding,
    HoopOutput,
    ClipOutput,
    VideoOutput,
    VideoArtifacts,
    KeyframeArtifact,
    ClipArtifacts,
    TagArtifact,
    WrappedClipOutput,
    BoxType,
    PointType,
    BatchSelectorFunctionType,
)
if torch.cuda.is_available():
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore
else:
    build_sam2_video_predictor = None
    print("No CUDA found. Cannot import SAM2.")
from .detectron_utils import (
    get_detectron2_skeleton_model,
    get_detectron2_hoop_model,
    filter_low_confidence_hoops,
)
from .solider_utils import get_solider_feature_extractor
from .x3d_utils import get_x3d_shot_model, x3d_infer_video, x3d_preprocess_video
from .editor_utils import create_thumbnail, create_highlight
from .possession_utils import get_possession_model, get_clip_vision_model
from .yolo_utils import (
    load_yolo_model,
    fetch_batch_of_frames,
    get_gpu_memory,
    calculate_optimal_batch_size,
    detect_balls_with_yolo,
)
from .dfine_utils import load_dfine_model, detect_balls_with_dfine
from .dino_utils import load_dino_model, detect_balls_with_dino
from .firebase_utils import (
    upload_files_to_firebase,
    upload_cluster_images_to_firebase,
    update_session_processing_progress,
    hardset_session_processing_progress,
    update_session_hoop_stats,
)
from .async_utils import (
    async_many_clip_inference_with_replicate,
    async_hoop_detection_with_replicate,
    async_many_clip_inference_with_cloudrun,
    async_hoop_detection_with_cloudrun,
)
from .utils import (
    get_checkpoints_dir,
    get_video_length,
    extract_subclip_ffmpeg,
    extract_subclip_ffmpeg_reencode_fps,
    get_video_frame_size,
    get_video_avg_fps,
    tick,
    sample_frames_from_video,
    move_subsampled_frames_out,
    merge_subsampled_frames_back,
    get_sampled_frame_names,
    show_mask,
    compute_bbox_iou,
    to_json,
    to_jsonlines,
    find_index_of_closest_timestamp,
    create_logger,
    fig2img,
    merge_video_segments,
    find_closest_detection_idx_to_point,
    find_closest_point_idx_to_timestamp,
    find_closest_box_idx_to_timestamp,
    unflatten_points,
    unflatten_boxes,
    label_collisions,
    remove_collisions,
    video_segments_to_video_boxes,
    generate_uuid,
    get_keyframe_timestamps,
    sample_frame_at_timestamp,
    truncate_clip_ffmpeg,
    load_image,
    format_bbox,
    plot_bbox,
)
from .ml_utils import (
    add_hoop_to_tracking_prompt,
    add_ball_to_tracking_prompt,
    add_person_to_tracking_prompt,
    init_inference_state,
    track_objects,
    select_person_keyframe,
    detect_persons,
    detect_shot_timestamps,
    assign_possessions,
    classify_possessions,
    detect_holder,
    find_first_possession_from_index,
    detect_shot_outcome,
    infer_person_keypoints,
    embed_player_trajectory,
    infer_clusters,
    get_shot_box_from_hoop_box,
    filter_detected_balls_pipeline,
    estimate_hoop_at_timestamp,
    score_ball_detections,
    score_ensemble_detections,
    smooth_detected_hoops,
    create_shot_classification_video,
)
from .utils import ACTION_SHOT, ACTION_ASSIST, ACTION_REBOUND


async def process_video(
    video_file: str,
    out_dir: str,
    paused_ts: Union[List[float], List[int]] = [],
    # Sample rate controls the processing speed
    frame_sample_rate: int = 1,
    context_window: int = 4,
    # Model hyperparameters
    min_detect_shot_score: float = 0.5,
    min_shot_score: float = 0.5,
    min_hoop_fast_score: float = 0.3,
    min_hoop_slow_score: float = 0.8,
    min_ball_score: float = 0.2,
    min_person_box_score: float = 0.1,
    min_person_keypoint_score: float = 0.1,
    min_segment_skeleton_iou: float = 0.6,
    hoop_fast_stride: int = 15,
    hoop_slow_stride: int = 300,
    dist_pixel_thres: int = 10,
    max_ball_person_ratio: int = 30,
    min_cluster_samples: int = 10,
    # Visualizes predictions on frames but slows processing
    visualize: bool = False,
    # Replicate parameters
    use_replicate: bool = False,
    replicate_api_token: Optional[str] = None,
    replicate_hoop_fast_model: Optional[str] = None,
    replicate_hoop_slow_model: Optional[str] = None,
    replicate_clip_model: Optional[str] = None,
    replicate_timeout: int = 10,
    # Cloud Run parameters
    use_cloudrun: bool = False,
    cloudrun_service_url: Optional[str] = None,
    cloudrun_invoker_sa: Optional[str] = None,
    cloudrun_timeout: int = 10,
    # Concurrency
    max_concurrent_tasks: int = 8,
    # Firebase parameters
    save_artifacts: bool = False,
    upload_firebase: bool = False,
    firebase_bucket: Optional[str] = None,
    session_id: Optional[str] = None,
    # Annotation
    annotated_hoop_points: Optional[Union[List[float], List[int]]] = None,
    override_hoop_boxes: Optional[Union[List[float], List[int]]] = None,
    skip_shot_idxs: List[int] = [],
    run_shot_idxs: List[int] = [],
    device_name: str = "cuda",
) -> VideoOutput:
    """Video inference pipeline.
    Given a video, find hoops across video using YOLO.
    Identify all potential shot timestamps using a frame diff heuristic.
    Define a "key frame" as the frame when a shot was identified.
    Find the ball closest to the rim using DFINE, and define it is a "key ball".
    Find all persons using Detectron2 person keypoints model.
    Use SAM2 to track the ball, the hoop, and all persons for N seconds before and after the shot keyframe.
    Identify the shot taker, the rebounder, and any assisters from N*2 clip.
    Cluster masks of actors across clips.

    Args:
        use_replicate: Do inference using Replicate (default: False)
        use_cloudrun: Do inference using Cloud Run service (default: False)

    Return:
        output: Predicted video outputs
    """
    # in practice, we would use the session id
    artifact_dir = join(out_dir, "artifacts")
    makedirs(artifact_dir, exist_ok=True)

    # Create logger for video inference
    log_path = join(artifact_dir, "trace.log")
    if isfile(log_path):
        remove(log_path)
    logger = create_logger("video_logger", log_path)

    if upload_firebase:
        assert firebase_bucket is not None, (
            "If uploading to firebase, the bucket must be provided"
        )
        start_time = tick()
        firebase_client = get_firebase_client()  # instantiate connection to firebase
        end_time = tick()
        logger.info(f"initialized firebase client - {end_time - start_time}s elapsed")
    else:
        firebase_client = None

    if not save_artifacts and upload_firebase:
        logging.warning(
            "`upload_firebase` set to true but `save_artfacts` set to false"
        )

    # Set progress to 0
    if upload_firebase and (session_id is not None):
        hardset_session_processing_progress(firebase_client, session_id, 0)
        logger.info("set progress to 0%")

    # Get video metadata
    video_width, video_height = get_video_frame_size(video_file)
    video_fps = int(round(get_video_avg_fps(video_file)))
    video_length = get_video_length(video_file)
    logger.info(
        f"found video (length={round(video_length, 2)}s,"
        f"fps={video_fps},width={video_width},height={video_height})"
    )

    # --- fast hoop inference ---

    start_time = tick()
    if use_replicate:
        assert replicate_api_token is not None, (
            "If using Replicate, the API token must be provided"
        )
        assert replicate_hoop_fast_model is not None, (
            "If using Replicate, fast hoop model must be provided"
        )
        assert replicate_clip_model is not None, (
            "If using Replicate, video model must be provided"
        )
        logger.info("[fast] kicking off replicate job for hoop detection")
        start_time = tick()
        result = await async_hoop_detection_with_replicate(
            replicate_api_token,
            replicate_hoop_fast_model,
            video_file,
            vid_stride=hoop_fast_stride,
            min_score=min_hoop_fast_score,
            annotated_points=annotated_hoop_points,
            override_boxes=override_hoop_boxes,
            replicate_timeout=replicate_timeout,
        )
        if isinstance(result, Exception):
            logger.error(f"[fast] task failed with exception: {result}")
            raise result  # can't do anything if fenceposts are empty
        if result.success:
            logger.info(
                f"[fast] successfully detected hoops - {result.elapsed_sec}s elapsed"
            )
        else:
            logger.error(
                f"[fast] failed to detect hoops - {result.error} - {result.elapsed_sec}s elapsed"
            )
        hoops = result.hoops
        frac_missing = result.frac_missing
        frac_moving = result.frac_moving
        assert hoops is not None, "hoops should not be None"
        assert frac_missing is not None, "frac_missing should not be None"
        assert frac_moving is not None, "frac_moving should not be None"
    elif use_cloudrun:
        assert cloudrun_service_url is not None, (
            "If using CloudRun, the service URL must be provided"
        )
        assert cloudrun_invoker_sa is not None, (
            "If using CloudRun, the invoker SA must be provided"
        )
        logger.info("[fast] kicking off cloud run job for hoop detection")
        start_time = tick()
        result = await async_hoop_detection_with_cloudrun(
            cloudrun_service_url,
            cloudrun_invoker_sa,
            video_file,
            vid_stride=hoop_fast_stride,
            min_score=min_hoop_fast_score,
            annotated_points=annotated_hoop_points,
            override_boxes=override_hoop_boxes,
            use_fast=True,
        )
        if isinstance(result, Exception):
            logger.error(f"[fast] task failed with exception: {result}")
            raise result  # can't do anything if fenceposts are empty
        if result.success:
            logger.info(
                f"[fast] successfully detected hoops - {result.elapsed_sec}s elapsed"
            )
        else:
            logger.error(
                f"[fast] failed to detect hoops - {result.error} - {result.elapsed_sec}s elapsed"
            )
        hoops = result.hoops
        frac_missing = result.frac_missing
        frac_moving = result.frac_moving
        assert hoops is not None, "hoops should not be None"
        assert frac_missing is not None, "frac_missing should not be None"
        assert frac_moving is not None, "frac_moving should not be None"
    else:
        yolo_checkpoint = join(
            get_checkpoints_dir(), "SHOT-Detection", "hoop.yolo11n.pt"
        )
        model = load_yolo_model(yolo_checkpoint, device_name=device_name)
        hoops, frac_missing, frac_moving = detect_hoops_fast(
            video_file=video_file,
            out_dir=out_dir,
            vid_stride=hoop_fast_stride,
            min_score=min_hoop_fast_score,
            preloaded_model=model,
            annotated_points=annotated_hoop_points,
            override_boxes=override_hoop_boxes,
            device_name=device_name,
        )
        del model
    end_time = tick()
    logger.info(
        f"[fast] predicted hoops in {len(hoops)} frames - {end_time - start_time}s elapsed"
    )

    # We won't stop processing based on these fractions but these may trigger frontend changes
    logger.info(f"[fast] fraction of frames missing hoop: {round(frac_missing, 3)}")
    logger.info(f"[fast] fraction of frames with moving hoop: {round(frac_moving, 3)}")

    # Update session with set
    if (
        upload_firebase
        and (session_id is not None)
        and (frac_missing is not None)
        and (frac_moving is not None)
    ):
        update_session_hoop_stats(
            firebase_client, session_id, frac_missing, frac_moving
        )
        logger.info("[fast] updated session doc with hoop fraction stats")

    # --- slow hoop inference (if needed) ---

    # Failing to find hoops or missing the hoop 50% of the time is a catastrophic failure
    if len(hoops) == 0 or frac_missing >= 0.5:
        start_time = tick()
        if use_replicate:
            assert replicate_api_token is not None, (
                "If using Replicate, the API token must be provided"
            )
            assert replicate_hoop_slow_model is not None, (
                "If using Replicate, slow hoop model must be provided"
            )
            assert replicate_clip_model is not None, (
                "If using Replicate, video model must be provided"
            )
            logger.info("[slow] kicking off replicate job for hoop detection")
            start_time = tick()
            result = await async_hoop_detection_with_replicate(
                replicate_api_token,
                replicate_hoop_slow_model,
                video_file,
                vid_stride=hoop_slow_stride,
                min_score=min_hoop_slow_score,
                annotated_points=annotated_hoop_points,
                override_boxes=override_hoop_boxes,
                replicate_timeout=replicate_timeout,
            )
            if isinstance(result, Exception):
                logger.error(f"[slow] task failed with exception: {result}")
                raise result  # can't do anything if fenceposts are empty
            if result.success:
                logger.info(
                    f"[slow] successfully detected hoops - {result.elapsed_sec}s elapsed"
                )
            else:
                logger.error(
                    f"[slow] failed to detect hoops - {result.error} - {result.elapsed_sec}s elapsed"
                )
            hoops = result.hoops
            frac_missing = result.frac_missing
            frac_moving = result.frac_moving
            assert hoops is not None, "hoops should not be None"
            assert frac_missing is not None, "frac_missing should not be None"
            assert frac_moving is not None, "frac_moving should not be None"
        elif use_cloudrun:
            assert cloudrun_service_url is not None, (
                "If using CloudRun, the service URL must be provided"
            )
            assert cloudrun_invoker_sa is not None, (
                "If using CloudRun, the invoker SA must be provided"
            )
            logger.info("[slow] kicking off cloud run job for hoop detection")
            start_time = tick()
            result = await async_hoop_detection_with_cloudrun(
                cloudrun_service_url,
                cloudrun_invoker_sa,
                video_file,
                vid_stride=hoop_fast_stride,
                min_score=min_hoop_fast_score,
                annotated_points=annotated_hoop_points,
                override_boxes=override_hoop_boxes,
                use_fast=False,
            )
            if isinstance(result, Exception):
                logger.error(f"[slow] task failed with exception: {result}")
                raise result  # can't do anything if fenceposts are empty
            if result.success:
                logger.info(
                    f"[slow] successfully detected hoops - {result.elapsed_sec}s elapsed"
                )
            else:
                logger.error(
                    f"[slow] failed to detect hoops - {result.error} - {result.elapsed_sec}s elapsed"
                )
            hoops = result.hoops
            frac_missing = result.frac_missing
            frac_moving = result.frac_moving
            assert hoops is not None, "hoops should not be None"
            assert frac_missing is not None, "frac_missing should not be None"
            assert frac_moving is not None, "frac_moving should not be None"
        else:
            model, _ = get_detectron2_hoop_model(device=device_name, batch_mode=True)
            hoops, frac_missing, frac_moving = detect_hoops_slow(
                video_file=video_file,
                out_dir=out_dir,
                vid_stride=hoop_slow_stride,
                min_score=min_hoop_slow_score,
                batch_size=8,
                preloaded_model=model,
                annotated_points=annotated_hoop_points,
                override_boxes=override_hoop_boxes,
                device_name=device_name,
            )
            del model
        end_time = tick()
        logger.info(
            f"[slow] predicted hoops in {len(hoops)} frames - {end_time - start_time}s elapsed"
        )

        # We won't stop processing based on these fractions but these may trigger frontend changes
        logger.info(f"[slow] fraction of frames missing hoop: {frac_missing}")
        logger.info(f"[slow] fraction of frames with moving hoop: {frac_moving}")

        # Update session with set
        if (
            upload_firebase
            and (session_id is not None)
            and (frac_missing is not None)
            and (frac_moving is not None)
        ):
            update_session_hoop_stats(
                firebase_client, session_id, frac_missing, frac_moving
            )
            logger.info("[slow] updated session doc with hoop fraction stats")

    if len(hoops) == 0:
        raise Exception("still no hoops found. quitting with error")

    # --- detect shot timestamps ---

    start_time = tick()
    logger.info(f"detecting shots -- min score: {min_detect_shot_score}")
    ball_key_ts_above = detect_shot_timestamps(
        video_file,
        video_length,
        hoops,
        box_padding_bottom=20,
        history_length=10,
        min_detect_shot_score=min_detect_shot_score,
        min_contiguous_length=3,
    )
    # Don't keep any shots within the first `context_window` seconds to ensure things are ok
    ball_key_ts_above = [ts for ts in ball_key_ts_above if ts >= context_window]
    end_time = tick()
    logger.info(
        f"identified {len(ball_key_ts_above)} key timestamps for ball -- above rim: {ball_key_ts_above}"
    )
    logger.info(
        f"identified {len(ball_key_ts_above)} candidate shots - {end_time - start_time}s elapsed"
    )

    if len(ball_key_ts_above) == 0:
        raise Exception("still shots found. quitting with error")

    # Set progress to 1 - just to show case some progress
    if upload_firebase and (session_id is not None):
        hardset_session_processing_progress(firebase_client, session_id, 1)
        logger.info("set progress to 1%")

    # --- clip inference for each detected shot ---

    shot_idxs: List[int] = []
    clip_outputs: List[ClipOutput] = []

    # Define a callback function to call after each task is done
    def task_callback(task: asyncio.Task[WrappedClipOutput]) -> None:
        """Callback function to call after each task is done.
        Used by both Replicate and cloudrun functions.

        Args:
            result: WrappedClipOutput
        """
        try:
            result = task.result()
            success = result.success
            clip_output = result.output
            elapsed_sec = result.elapsed_sec
            shot_idx = result.shot_idx
            if success and (clip_output is not None):
                logger.info(
                    f"successfully processed shot {shot_idx} - {elapsed_sec}s elapsed"
                )
            else:
                logger.error(
                    f"failed to process shot {shot_idx} - empty output - {elapsed_sec}s elapsed"
                )
        except Exception as e:
            logger.error(f"failed to get result for shot - {e}")

        # Regardless of success or failure, update the session processing progress
        if upload_firebase and (session_id is not None):
            update_session_processing_progress(
                firebase_client, session_id, 100 * (1 / len(ball_key_ts_above))
            )
            logger.info(
                f"incremented progress by {100 * (1 / len(ball_key_ts_above))}%"
            )

    if use_replicate:
        assert replicate_api_token is not None, (
            "If using Replicate, the API token must be provided"
        )
        assert replicate_clip_model is not None, (
            "If using Replicate, video model must be provided"
        )
        # Send many requests to Replicate to process these clips
        start_time = tick()
        logger.info(f"kicking off {len(ball_key_ts_above)} replicate jobs for clips")
        results = await async_many_clip_inference_with_replicate(
            replicate_api_token,
            replicate_clip_model,
            video_file,
            video_length,
            hoops,
            ball_key_ts_above,
            firebase_bucket=firebase_bucket,
            context_window=context_window,
            min_shot_score=min_shot_score,
            min_ball_score=min_ball_score,
            min_person_box_score=min_person_box_score,
            min_person_keypoint_score=min_person_keypoint_score,
            min_segment_skeleton_iou=min_segment_skeleton_iou,
            dist_pixel_thres=dist_pixel_thres,
            max_ball_person_ratio=max_ball_person_ratio,
            frame_sample_rate=frame_sample_rate,
            replicate_timeout=replicate_timeout,
            max_concurrent_tasks=max_concurrent_tasks,
            task_callback=task_callback,
        )
        end_time = tick()
        for result in results:  # inform user about any exceptions
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
        # remove any exceptions and sort by shot index
        results = [r for r in results if not isinstance(r, Exception)]
        results = sorted(results, key=lambda x: x.shot_idx)
        # Aggregate all the embeddings across the replicate jobs
        for i in range(len(results)):
            if results[i].success:
                clip_output = results[i].output
                shot_idx = results[i].shot_idx
                if clip_output is not None:
                    clip_outputs.append(clip_output)
                    shot_idxs.append(shot_idx)
    elif use_cloudrun:
        assert cloudrun_service_url is not None, (
            "If using CloudRun, the service URL must be provided"
        )
        assert cloudrun_invoker_sa is not None, (
            "If using CloudRun, the invoker SA must be provided"
        )
        # Send many requests to Replicate to process these clips
        start_time = tick()
        logger.info(f"kicking off {len(ball_key_ts_above)} cloud run jobs for clips")
        results = await async_many_clip_inference_with_cloudrun(
            cloudrun_service_url,
            cloudrun_invoker_sa,
            video_file,
            video_length,
            hoops,
            ball_key_ts_above,
            firebase_bucket=firebase_bucket,
            context_window=context_window,
            min_shot_score=min_shot_score,
            min_ball_score=min_ball_score,
            min_person_box_score=min_person_box_score,
            min_person_keypoint_score=min_person_keypoint_score,
            min_segment_skeleton_iou=min_segment_skeleton_iou,
            dist_pixel_thres=dist_pixel_thres,
            max_ball_person_ratio=max_ball_person_ratio,
            frame_sample_rate=frame_sample_rate,
            cloudrun_timeout=cloudrun_timeout,
            max_concurrent_tasks=max_concurrent_tasks,
            task_callback=task_callback,
        )
        end_time = tick()
        for result in results:  # inform user about any exceptions
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
        # remove any exceptions and sort by shot index
        results = [r for r in results if not isinstance(r, Exception)]
        results = sorted(results, key=lambda x: x.shot_idx)
        # Aggregate all the embeddings across the replicate jobs
        for i in range(len(results)):
            if results[i].success:
                clip_output = results[i].output
                shot_idx = results[i].shot_idx
                if clip_output is not None:
                    clip_outputs.append(clip_output)
                    shot_idxs.append(shot_idx)
    else:
        # Load all models
        preloaded_models = load_clip_models(device_name=device_name)
        # Find candidates to click for shows below the net/rim
        # For each timestamp, we will pull the key frame, and sample frame around it
        for i, ball_key_ts in enumerate(ball_key_ts_above):
            if i in skip_shot_idxs:  # helper to skip some shots
                continue
            if len(run_shot_idxs) > 0 and (i not in run_shot_idxs):
                continue

            start_ts = round(max(0, ball_key_ts - context_window), 3)
            end_ts = round(min(video_length, ball_key_ts + context_window), 3)
            logger.info(
                f"begin processing shot {i} - start {start_ts}s, end: {end_ts}s, keyframe: {ball_key_ts}s"
            )

            start_time = tick()
            try:
                clip_output = infer_clip(
                    video_file,
                    i,
                    # Find the subset of hoops that are useful for this shot
                    [hoop for hoop in hoops if start_ts <= hoop.ts <= end_ts],
                    start_ts,
                    end_ts,
                    ball_key_ts,
                    join(out_dir, f"clip-{i}"),
                    min_shot_score=min_shot_score,
                    min_ball_score=min_ball_score,
                    min_person_keypoint_score=min_person_keypoint_score,
                    min_segment_skeleton_iou=min_segment_skeleton_iou,
                    dist_pixel_thres=dist_pixel_thres,
                    max_ball_person_ratio=max_ball_person_ratio,
                    frame_sample_rate=frame_sample_rate,
                    visualize=visualize,
                    save_artifacts=save_artifacts,
                    upload_firebase=upload_firebase,
                    firebase_bucket=join(firebase_bucket, f"clip-{i}")
                    if firebase_bucket
                    else None,
                    preloaded_models=preloaded_models,
                    device_name=device_name,
                )
            except Exception as e:
                logger.error(f"failed to process {i} - exception found - {e}")
                raise e  # raise it since we are testing locally

            end_time = tick()
            if clip_output is not None:
                logger.info(
                    f"successfully processed shot {i} - {end_time - start_time}s elapsed"
                )
            else:
                # We do not use `logger.error` to avoid sentry but still record the failure
                logger.info(
                    f"failed to process shot {i} - empty output - {end_time - start_time}s elapsed"
                )

            if clip_output is not None:
                clip_outputs.append(clip_output)
                shot_idxs.append(i)

            if upload_firebase and (session_id is not None):
                update_session_processing_progress(
                    firebase_client, session_id, 100 * (1 / len(ball_key_ts_above))
                )
                logger.info(
                    f"shot {i} - incremented progress by {100 * (1 / len(ball_key_ts_above))}%"
                )

    # --- cluster embeddings w/ projected dbscan ---

    if len(clip_outputs) > 0:
        start_time = tick()
        all_embeddings = [c.embeddings for c in clip_outputs if c is not None]
        clusters, cluster_classes = infer_clusters(
            shot_idxs,
            all_embeddings,
            logger=logger,
            min_samples=min_cluster_samples,
        )
        end_time = tick()
        logger.info(
            f"created {len(clusters)} clusters, classes: {cluster_classes} - {end_time - start_time}s elapsed"
        )
    else:
        clusters = []
        cluster_classes = []
        logger.error("found no clip outputs, punting on clusters")

    artifacts = VideoArtifacts()
    if save_artifacts:
        # --- save cluster images to firebase ---

        if upload_firebase and (firebase_bucket is not None) and (len(clusters) > 0):
            # Upload the clusters to firebase - this will also assign `image_url` to the cluster objects
            start_time = tick()
            clusters = upload_cluster_images_to_firebase(
                clusters, out_dir, firebase_bucket
            )
            end_time = tick()
            logger.info(
                f"uploaded {len(clusters)} cluster images to firebase {firebase_bucket} - {end_time - start_time}s elapsed"
            )

        # --- save metadata, video preview, and hoop predictions to firebase ---

        start_time = tick()
        # Save video level outputs
        metadata_path = join(artifact_dir, "metadata.json")
        hoops_path = join(artifact_dir, "hoops.jsonl")
        preview_path = join(artifact_dir, "preview.jpg")  # preview for full video
        cluster_path = join(artifact_dir, "clusters.jsonl")
        create_thumbnail(video_file, preview_path)  # create preview
        # Need to use dict() instead of model_dump() for pydantic v1 (required by cog)
        to_jsonlines([cl.dict() for cl in clusters], cluster_path)
        metadata = {
            "args": {
                "video_file": video_file,
                "paused_ts": paused_ts,
                "out_dir": out_dir,
                "frame_sample_rate": frame_sample_rate,
                "context_window": context_window,
                "min_detect_shot_score": min_detect_shot_score,
                "min_shot_score": min_shot_score,
                "min_hoop_fast_score": min_hoop_fast_score,
                "min_hoop_slow_score": min_hoop_slow_score,
                "min_ball_score": min_ball_score,
                "min_person_box_score": min_person_box_score,
                "min_person_keypoint_score": min_person_keypoint_score,
                "min_segment_skeleton_iou": min_segment_skeleton_iou,
                "hoop_fast_stride": hoop_fast_stride,
                "hoop_slow_stride": hoop_slow_stride,
                "dist_pixel_thres": dist_pixel_thres,
                "max_ball_person_ratio": max_ball_person_ratio,
                "min_cluster_samples": min_cluster_samples,
                "save_artifacts": save_artifacts,
                "annotated_hoop_points": annotated_hoop_points,
                "override_hoop_boxes": override_hoop_boxes,
                "skip_shot_idxs": skip_shot_idxs,
                "run_shot_idxs": run_shot_idxs,
                "device_name": device_name,
            },
            "video_length": video_length,
            "ball_key_ts_above": ball_key_ts_above,
        }
        to_json(metadata, metadata_path)
        # Need to use dict() instead of model_dump() for pydantic v1 (required by cog)
        to_jsonlines([hoop.dict() for hoop in hoops], hoops_path)
        end_time = tick()
        logger.info(f"saving outputs - {end_time - start_time}s elapsed")
        if upload_firebase and (firebase_bucket is not None):
            # Upload artifacts to firebase
            start_time = tick()
            artifact_names = [
                "trace.log",
                "fenceposts.log",
                "metadata.json",
                "fenceposts.pt",
                "clusters.jsonl",
                "preview.jpg",
            ]
            artifact_urls = upload_files_to_firebase(
                artifact_names, artifact_dir, firebase_bucket
            )
            artifacts = VideoArtifacts(**artifact_urls)
            end_time = tick()
            logger.info(
                f"uploaded {len(artifact_names)} artifacts to firebase {firebase_bucket} - {end_time - start_time}s elapsed"
            )

    if upload_firebase and (session_id is not None):
        hardset_session_processing_progress(firebase_client, session_id, 100)
        logger.info("set progress to 100%")

    if len(clip_outputs) == 0:
        logger.error("found 0 successfully processed clips; quitting...")

    output = {
        "fps": video_fps,
        "width": video_width,
        "height": video_height,
        "length": video_length,
        "firebase_bucket": firebase_bucket,
        "clips": clip_outputs,
        "clusters": clusters,
        "cluster_classes": cluster_classes,
        "artifacts": artifacts,
    }
    output = VideoOutput(**output)
    return output


def detect_hoops_fast(
    video_file: str,
    out_dir: str,
    vid_stride: int = 15,
    min_score: float = 0.3,
    preloaded_model: Optional[Any] = None,
    annotated_points: Optional[Union[List[float], List[int]]] = None,
    override_boxes: Optional[Union[List[float], List[int]]] = None,
    device_name: str = "cuda",
) -> Tuple[List[HoopOutput], float, float]:
    """Fast hoop inference pipeline.
    This samples every 0.5s from the video and runs hoop detection using YOLO.
    @return hoops - detected hoops
    @return frac_movement - percentage of frames with movement
    @return frac_missing - percentage of frames with missing data
    """
    if preloaded_model is not None:
        model = preloaded_model
    else:
        yolo_checkpoint = join(
            get_checkpoints_dir(), "SHOT-Detection", "hoop.yolo11n.pt"
        )
        model = load_yolo_model(yolo_checkpoint, device_name=device_name)

    # Save stuff to this directory
    artifact_dir = join(out_dir, "artifacts")
    makedirs(artifact_dir, exist_ok=True)

    # Separate logger for the fencepost inference
    logger = create_logger("fast_hoop_logger", join(artifact_dir, "hoops-fast.log"))

    # Implement video cropping logic here
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video {video_file}")

    # Get video properties
    video_length = round(float(get_video_length(video_file)), 3)
    logger.info(f"Found video with length {video_length}")

    # Get inference parameters
    device_memory = get_gpu_memory()
    batch_size = calculate_optimal_batch_size((640, 640, 3), device_memory)
    logger.info(
        f"Found {round(device_memory, 3)} device memory, setting batch_size={batch_size}"
    )

    # Keep around the last hoop
    last_hoop = None

    # This is the output that will be returned
    outputs: List[HoopOutput] = []
    num_frames_missing = 0
    num_total = 0

    start_time = tick()
    pbar = tqdm(total=video_length)
    while cap.isOpened():
        batch_frames, batch_timestamps = fetch_batch_of_frames(
            cap, batch_size, vid_stride
        )
        assert len(batch_frames) == len(batch_timestamps)
        if len(batch_frames) == 0:
            break

        if override_boxes is not None:
            # If there is an override provided, use it find the right box
            candidate_boxes = unflatten_boxes(override_boxes)
            for timestamp_i in batch_timestamps:
                closest_idx = find_closest_box_idx_to_timestamp(
                    candidate_boxes, timestamp_i
                )
                box_i = candidate_boxes[closest_idx][:4]

                output_i = HoopOutput(ts=round(timestamp_i, 3), prob=1, box=box_i)
                outputs.append(output_i)
        else:
            # Perform batched inference
            results = model(
                batch_frames,
                conf=min_score,
                imgsz=1024,  # important to detect hoops properly
                half=True,
                verbose=False,
                device=device_name,
            )
            for i in range(len(results)):
                timestamp_i = batch_timestamps[i]
                # If there are hoops in the frame
                has_hoop_i = len(results[i].boxes) > 0
                # Find the annotated hoop point for this frame if one exists
                if has_hoop_i:
                    if annotated_points is not None:
                        candidate_points = unflatten_points(annotated_points)
                        closest_idx = find_closest_point_idx_to_timestamp(
                            candidate_points, timestamp_i
                        )
                        point = candidate_points[closest_idx][:2]
                        # Find the one closest to the annotated point
                        best_idx = find_closest_detection_idx_to_point(
                            results[i].boxes.xyxy.cpu(), point
                        )
                    elif last_hoop is not None:
                        # Find the one closest to the last hoop
                        point: PointType = [
                            int((last_hoop[0] + last_hoop[2]) / 2),
                            int((last_hoop[1] + last_hoop[3]) / 2),
                        ]
                        # Find the one closest to the annotated point
                        best_idx = find_closest_detection_idx_to_point(
                            results[i].boxes.xyxy.cpu(), point
                        )
                    else:
                        # Take the box with the biggest confidence score
                        best_idx = results[i].boxes.conf.argmax().item()

                    # Find the best hoop index
                    box_i = format_bbox(
                        results[i].boxes[best_idx].xyxy[0].cpu().tolist()
                    )
                    prob_i = round(float(results[i].boxes[best_idx].conf.item()), 3)

                    # Save found hoop as the last hoop
                    last_hoop = box_i
                else:
                    # If no hoop, then assume its the same as the last known frame
                    box_i = last_hoop
                    prob_i = None
                    num_frames_missing += 1

                # If we cannot find a hoop, and there is no prior hoop, then skip this frame entirely
                # Note that his means |outputs| != |timestamps|
                if box_i is None:
                    num_frames_missing += 1
                    continue

                output_i = HoopOutput(ts=round(timestamp_i, 3), prob=prob_i, box=box_i)
                outputs.append(output_i)

        num_total += len(batch_timestamps)
        pbar.n = max(batch_timestamps)
        pbar.refresh()
    cap.release()
    pbar.close()

    end_time = tick()
    logger.info(
        f"detected hoops on {len(outputs)} frames - {end_time - start_time}s elapsed"
    )

    start_time = tick()
    outputs, num_frames_moving = smooth_detected_hoops(outputs)
    end_time = tick()
    logger.info(
        f"smoothed hoops on {len(outputs)} frames - {end_time - start_time}s elapsed"
    )

    # Compute fractions
    frac_missing = num_frames_missing / float(num_total)
    frac_moving = num_frames_moving / float(num_total)

    return outputs, frac_missing, frac_moving


def detect_hoops_slow(
    video_file: str,
    out_dir: str,
    vid_stride: int = 300,
    min_score: float = 0.8,
    batch_size: int = 32,
    preloaded_model: Optional[Any] = None,
    annotated_points: Optional[Union[List[float], List[int]]] = None,
    override_boxes: Optional[Union[List[float], List[int]]] = None,
    device_name: str = "cuda",
) -> Tuple[List[HoopOutput], float, float]:
    """Slow hoop inference pipeline.
    This samples every 10s from the video and runs hoop detection using Detectron2.
    @return hoops - detected hoops
    @return frac_movement - percentage of frames with movement
    @return frac_missing - percentage of frames with missing data
    """
    if preloaded_model is not None:
        model = preloaded_model
    else:
        model, _ = get_detectron2_hoop_model(device=device_name, batch_mode=True)

    video_width, video_height = get_video_frame_size(video_file)
    video_fps = int(round(get_video_avg_fps(video_file)))

    artifact_dir = join(out_dir, "artifacts")
    makedirs(artifact_dir, exist_ok=True)

    # Separate logger for the fencepost inference
    logger = create_logger("slow_hoop_logger", join(artifact_dir, "hoops-slow.log"))

    # Sample frames rather than using cv2 because we toss so many away
    start_time = tick()
    frame_dir = join(out_dir, "hoops-slow")
    makedirs(frame_dir, exist_ok=True)
    sample_frames_from_video(video_file, frame_dir, stride=vid_stride, start=vid_stride)
    frame_names = get_sampled_frame_names(frame_dir)
    frame_paths = [join(frame_dir, frame_name) for frame_name in frame_names]
    frame_times = [vid_stride / video_fps * (i + 1) for i in range(len(frame_paths))]
    num_frames = len(frame_names)
    end_time = tick()
    logger.info(
        f"identified {len(frame_names)} fenceposts - {end_time - start_time}s elapsed"
    )

    # Keep around the last hoop
    last_hoop = None

    # This is the output that will be returned
    outputs: List[HoopOutput] = []
    num_frames_missing = 0
    num_total = 0

    start_time = tick()
    pbar = tqdm(total=num_frames)
    if override_boxes is not None:
        # If there is an override provided, use it
        candidate_boxes = unflatten_boxes(override_boxes)
        for i in range(num_frames):
            closest_idx = find_closest_box_idx_to_timestamp(
                candidate_boxes, frame_times[i]
            )
            box_i = candidate_boxes[closest_idx][:4]

            output_i = HoopOutput(ts=round(frame_times[i], 3), prob=1, box=box_i)
            outputs.append(output_i)

            num_total += 1
            pbar.update()
    else:
        selector_fn: Optional[BatchSelectorFunctionType] = None

        # If we have a user annotated found, use it to pick
        if annotated_points is not None:
            candidate_points = unflatten_points(annotated_points)

            def closest_to_annotation(boxes: torch.Tensor, batch_idx: int) -> int:
                closest_idx = find_closest_point_idx_to_timestamp(
                    candidate_points, frame_times[batch_idx]
                )
                point = candidate_points[closest_idx][:2]
                best_idx = find_closest_detection_idx_to_point(boxes, point)
                return best_idx

            selector_fn = closest_to_annotation
        else:
            selector_fn = None

        num_batches = len(frame_paths) // batch_size + (
            len(frame_paths) % batch_size != 0
        )
        for i in range(num_batches):
            start_idx, end_idx = batch_size * i, batch_size * (i + 1)
            batch_paths = frame_paths[start_idx:end_idx]
            batch_bgr = np.stack([load_image(path) for path in batch_paths])

            with torch.no_grad():
                batch_preds = model(batch_bgr)

            for j in range(len(batch_preds)):
                cur_idx = start_idx + j
                timestamp_j = frame_times[cur_idx]
                preds = batch_preds[j]
                preds = filter_low_confidence_hoops(preds, min_score=min_score)

                has_hoop_j = len(preds["instances"]) > 0
                if has_hoop_j:
                    boxes = preds["instances"].pred_boxes.tensor.cpu()
                    if selector_fn is not None:
                        idx = selector_fn(boxes, cur_idx)
                    else:
                        idx = 0
                    box_j = format_bbox(boxes[idx].numpy().tolist())
                    prob_j = round(float(preds["instances"].scores[idx].item()), 3)

                    # Save found hoop as the last hoop
                    last_hoop = box_j
                else:
                    box_j = last_hoop
                    prob_j = None
                    num_frames_missing += 1

                # If we cannot find a hoop, and there is no prior hoop, then skip this frame entirely
                # Note that his means |outputs| != |timestamps|
                if box_j is None:
                    num_frames_missing += 1
                    continue

                num_total += 1
                output_i = HoopOutput(ts=round(timestamp_j, 3), prob=prob_j, box=box_j)
                outputs.append(output_i)

            pbar.update(len(batch_paths))
    pbar.close()

    end_time = tick()
    logger.info(
        f"detected hoops on {num_frames} frames - {end_time - start_time}s elapsed"
    )

    # Estimate the movement from boxes
    start_time = tick()
    outputs, num_frames_moving = smooth_detected_hoops(outputs)
    end_time = tick()
    logger.info(
        f"smoothed hoops on {len(outputs)} frames - {end_time - start_time}s elapsed"
    )

    # Compute fractions
    frac_missing = num_frames_missing / float(num_total)
    frac_moving = num_frames_moving / float(num_total)

    return outputs, frac_missing, frac_moving


def load_clip_models(
    device_name: str = "cuda",
) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    r"""Load all models needed for video inference in memory.
    :param device_name: GPU hardware device (default: cuda)
    :return: all models
    """
    assert build_sam2_video_predictor is not None
    device = torch.device(device_name)

    # Load DFine for tracking balls
    dfine_model, dfine_processor = load_dfine_model(
        "ustc-community/dfine-xlarge-obj365", device_name=device_name
    )
    dfine_model = dfine_model.to(device)

    # Load Grounding DINO for backup ball detection
    dino_model, dino_processor = load_dino_model(
        "IDEA-Research/grounding-dino-base", device_name=device_name
    )

    # Load YOLO for backup backup ball detection
    yolo_checkpoint = join(get_checkpoints_dir(), "SHOT-Detection", "ball.yolo11x.pt")
    yolo_model = load_yolo_model(yolo_checkpoint, device_name=device_name)

    # Load SAM2
    sam_checkpoint = join(
        get_checkpoints_dir(), "SAM2-InstanceSegmentation", "sam2.1_hiera_large.pt"
    )
    assert isfile(sam_checkpoint), f"SAM2 checkpoint {sam_checkpoint} not found"
    sam_config = "configs/sam2.1/sam2.1_hiera_l.yaml"  # keep fixed
    sam_model = build_sam2_video_predictor(sam_config, sam_checkpoint, device=device)

    # Load slow skeleton model
    person_keypoints_model, _ = get_detectron2_skeleton_model(
        device=device_name, batch_mode=True
    )

    # Load feature extractor
    embed_checkpoint = join(
        get_checkpoints_dir(), "PERSON-Tracking", "swin_base_msmt17.pth"
    )
    embed_model = get_solider_feature_extractor(embed_checkpoint, device=device_name)

    # Load shot classifier
    x3d_checkpoint = join(
        get_checkpoints_dir(), "SHOT-Classification", "x3d_m.04.21.pt"
    )
    assert isfile(x3d_checkpoint), f"X3D checkpoint {x3d_checkpoint} not found"
    shot_model = get_x3d_shot_model(model_name="x3d_m", weight_file=x3d_checkpoint)
    shot_model = shot_model.to(device)  # type: ignore

    # Load possession classifier
    possession_model = get_possession_model()
    possession_model = possession_model.to(device)

    # Load image embedder for possession model
    possession_embedder = get_clip_vision_model()
    possession_embedder = possession_embedder.to(device)  # type: ignore

    return (
        shot_model,
        dfine_processor,
        dfine_model,
        dino_processor,
        dino_model,
        yolo_model,
        sam_model,
        person_keypoints_model,
        embed_model,
        possession_model,
        possession_embedder,
    )


def infer_clip(
    video_file: str,
    shot_idx: int,
    hoops: List[HoopOutput],
    start_ts: float,
    end_ts: float,
    ball_key_ts: float,
    shot_dir: str,
    frame_sample_rate: int = 1,
    min_shot_score: float = 0.5,
    min_ball_score: float = 0.2,
    min_person_box_score: float = 0.1,
    min_person_keypoint_score: float = 0.1,
    min_segment_skeleton_iou: float = 0.6,
    dist_pixel_thres: int = 10,
    max_ball_person_ratio: int = 30,
    preloaded_models: Optional[Tuple] = None,
    visualize: bool = False,
    save_artifacts: bool = False,
    upload_firebase: bool = False,
    firebase_bucket: Optional[str] = None,
    device_name: str = "cuda",
) -> Optional[ClipOutput]:
    """Shot clip inference pipeline.
    @note: we purposefully make this an independent function that loads models separately so we can deploy this to replicate.
    @note: if we want to create artifacts from inference, set `save_artifacts` to True.
    @note: if we want to upload artifacts to firebase, set `upload_firebase` to True, and provide the `firebase_bucket` path.
    @note: if you have preloaded models, use `preloaded_models`
    @return: did the job fail prematurly?
    """
    fn_start_time = tick()  # use this as a global meter

    artifact_dir = join(shot_dir, "artifacts")
    makedirs(artifact_dir, exist_ok=True)

    # separate logger for the shot
    log_path = join(artifact_dir, "trace.log")
    if isfile(log_path):
        remove(log_path)
    logger = create_logger(f"shot_logger_{shot_idx}", log_path)

    if len(hoops) == 0:
        logger.error("no hoops found, skipping shot")
        return None

    if upload_firebase:
        assert firebase_bucket is not None, (
            "If uploading to firebase, the bucket must be provided"
        )
        start_time = tick()
        get_firebase_client()  # instantiate connection to firebase - needed to upload files
        end_time = tick()
        logger.info(
            f"initialized firebase client - {end_time - start_time}s elapsed - {end_time - fn_start_time} total"
        )

    # --- Load the models ---

    if preloaded_models is not None:
        assert len(preloaded_models) == 11, (
            f"Expected 11 models and processors. Got {len(preloaded_models)}."
        )
        models = preloaded_models
    else:
        start_time = tick()
        models = load_clip_models(device_name=device_name)
        end_time = tick()
        logger.info(
            f"loaded models - {end_time - start_time}s elapsed - {end_time - fn_start_time} total"
        )

    video_width, video_height = get_video_frame_size(video_file)
    raw_fps = int(round(get_video_avg_fps(video_file)))
    (
        shot_model,
        dfine_processor,
        dfine_model,
        dino_processor,
        dino_model,
        yolo_model,
        sam_model,
        person_keypoints_model,
        embed_model,
        possession_model,
        possession_embedder,
    ) = models

    # --- Extract the subclip & format it to be 30 FPS ---

    start_time = tick()
    clip_fps = 30
    clip_path = join(shot_dir, "clip.mp4")
    if (clip_fps - 1) < raw_fps < (clip_fps + 1):
        # This is close enough and we do not need to reencode
        extract_subclip_ffmpeg(video_file, start_ts, end_ts, clip_path)
    else:
        # We need to reencode to be 30 FPS
        extract_subclip_ffmpeg_reencode_fps(
            video_file, start_ts, end_ts, clip_path, out_fps=clip_fps
        )
    end_time = tick()
    logger.info(
        f"extracted subclip ({round(start_ts, 2)} to {round(end_ts, 2)}) to {clip_path} - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total",
    )

    # --- Sample clip into frames ---

    all_frame_dir = join(shot_dir, "frames-all")
    makedirs(all_frame_dir, exist_ok=True)
    start_time = tick()
    sample_stride = max(int(frame_sample_rate), 1)
    # Sample every frame (which we need to do for highlight creation)
    sample_frames_from_video(clip_path, all_frame_dir, stride=1)
    all_frame_names = get_sampled_frame_names(all_frame_dir)
    end_time = tick()
    logger.info(
        f"sampled {len(all_frame_names)} frames to {all_frame_dir} - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total",
    )

    # Move and rename a subset of the frames
    frame_dir = join(shot_dir, "frames")
    makedirs(frame_dir, exist_ok=True)
    start_time = tick()
    # These are the frames we will use to actually run inference
    move_subsampled_frames_out(all_frame_dir, frame_dir, stride=sample_stride)
    frame_names = get_sampled_frame_names(frame_dir)
    frame_paths = [join(frame_dir, frame_name) for frame_name in frame_names]
    end_time = tick()
    logger.info(
        f"moved {len(frame_names)} frames to {frame_dir} - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )

    # Assign an index for each frame sorted by frame name (0 -> |frame_names| - 1)
    frame_idxs = list(range(len(frame_names)))

    # Estimate timestamps from frames
    frame_ts = np.linspace(start_ts, end_ts, len(frame_names)).tolist()
    frame_ts = [round(ts, 2) for ts in frame_ts]

    # --- Ball detection ---

    # Pick out a guess for where the ball is (this is when we detected it for the shot)
    ball_candidate_idx = find_index_of_closest_timestamp(frame_ts, ball_key_ts)

    # When we do ball inference, we want to try to use several frames
    offsets = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    ball_candidate_idxs = []
    ball_candidate_times = []
    ball_candidate_paths = []
    for offset in offsets:
        if ball_candidate_idx + offset < 0:
            continue
        if ball_candidate_idx + offset >= max(frame_idxs):
            continue
        offset_idx = ball_candidate_idx + offset
        ball_candidate_idxs.append(offset_idx)
        ball_candidate_times.append(frame_ts[offset_idx])
        ball_candidate_paths.append(join(frame_dir, frame_names[offset_idx]))
    hoop_candidates = [
        estimate_hoop_at_timestamp(hoops, ts) for ts in ball_candidate_times
    ]
    num_candidates = len(ball_candidate_idxs)

    # Run detections with DFINE
    start_time = tick()
    balls_pred_dfine_ = detect_balls_with_dfine(
        dfine_model,
        dfine_processor,
        ball_candidate_paths,
        min_conf=min_ball_score,
        device_name=device_name,
    )
    balls_pred_dfine = filter_detected_balls_pipeline(
        ball_candidate_times,
        balls_pred_dfine_,
        hoop_candidates,
        video_height,
        video_width,
    )
    end_time = tick()
    logger.info(
        f"[model=DFINE,conf={min_ball_score}] "
        f"detected {[len(balls) for balls in balls_pred_dfine_]} ball(s) - "
        f"filtered {[len(balls) for balls in balls_pred_dfine]} ball(s) - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )
    # Run detections with DINO
    start_time = tick()
    balls_pred_dino_ = detect_balls_with_dino(
        dino_model,
        dino_processor,
        ball_candidate_paths,
        box_threshold=min_ball_score,
        device_name=device_name,
    )
    balls_pred_dino = filter_detected_balls_pipeline(
        ball_candidate_times,
        balls_pred_dino_,
        hoop_candidates,
        video_height,
        video_width,
    )
    end_time = tick()
    logger.info(
        f"[model=DINO,conf={min_ball_score}] "
        f"detected {[len(balls) for balls in balls_pred_dino_]} ball(s) - "
        f"filtered {[len(balls) for balls in balls_pred_dino]} ball(s) - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )
    # Run detections with YOLO
    start_time = tick()
    balls_pred_yolo_ = detect_balls_with_yolo(
        yolo_model,
        ball_candidate_paths,
        hoop_candidates,
        min_conf=min_ball_score,
        crop_size=256,
        device_name=device_name,
    )
    balls_pred_yolo = filter_detected_balls_pipeline(
        ball_candidate_times,
        balls_pred_yolo_,
        hoop_candidates,
        video_height,
        video_width,
    )
    end_time = tick()
    logger.info(
        f"[model=YOLO,conf={min_ball_score}] "
        f"detected {[len(balls) for balls in balls_pred_yolo_]} ball(s) - "
        f"filtered {[len(balls) for balls in balls_pred_yolo]} ball(s) - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )

    # --- Score detected balls ---

    score_tuples: List[Tuple[str, int, float, BoxType]] = []
    for i in range(num_candidates):
        # Ensemble scores measure agreement between models
        dfine_ensemble_score_i = score_ensemble_detections(
            balls_pred_dfine[i],
            [balls_pred_dino[i], balls_pred_yolo[i]],
            min_iou=0.5,
        )
        dino_ensemble_score_i = score_ensemble_detections(
            balls_pred_dino[i],
            [balls_pred_dfine[i], balls_pred_yolo[i]],
            min_iou=0.5,
        )
        yolo_ensemble_score_i = score_ensemble_detections(
            balls_pred_yolo[i],
            [balls_pred_dfine[i], balls_pred_dino[i]],
            min_iou=0.5,
        )
        # Individual scores measure how "ideal" the detection is
        dfine_indiv_scores_i = score_ball_detections(
            balls_pred_dfine[i],
            hoop_candidates[i].box,
            video_height,
            apex_weight=0.5,
            conf_weight=0.5,
        )
        dino_indiv_scores_i = score_ball_detections(
            balls_pred_dino[i],
            hoop_candidates[i].box,
            video_height,
            apex_weight=0.5,
            conf_weight=0.5,
        )
        yolo_indiv_scores_i = score_ball_detections(
            balls_pred_yolo[i],
            hoop_candidates[i].box,
            video_height,
            apex_weight=0.5,
            conf_weight=0.5,
        )
        # Compute aggregate scores
        for j in range(len(balls_pred_dfine[i])):
            dfine_score_j = dfine_ensemble_score_i[j] * dfine_indiv_scores_i[j]
            score_tuples.append(("dfine", i, dfine_score_j, balls_pred_dfine[i][j]))
        for j in range(len(balls_pred_dino[i])):
            dino_score_j = dino_ensemble_score_i[j] * dino_indiv_scores_i[j]
            score_tuples.append(("dino", i, dino_score_j, balls_pred_dino[i][j]))
        for j in range(len(balls_pred_yolo[i])):
            yolo_score_j = yolo_ensemble_score_i[j] * yolo_indiv_scores_i[j]
            score_tuples.append(("yolo", i, yolo_score_j, balls_pred_yolo[i][j]))

    if len(score_tuples) == 0:
        logger.error("no ball found. quitting with error")
        return None

    # Find the best detection
    score_tuples = sorted(score_tuples, key=lambda x: x[2])[::-1]
    for i in range(min(len(score_tuples), 5)):
        logger.info(
            f"rank {i + 1}: model={score_tuples[i][0]} "
            f"index={score_tuples[i][1]} score={round(score_tuples[i][2], 3)} "
            f"box={score_tuples[i][3]}"
        )

    if score_tuples[0][2] == 0:
        logger.error("no ball found. quitting with error")
        return None

    # --- Pick the best ball or quit ---

    ball_keyframe_idx = ball_candidate_idxs[score_tuples[0][1]]
    ball_keyframe_ts = frame_ts[ball_keyframe_idx]
    ball_keyframe_path = frame_paths[ball_keyframe_idx]
    ball_keyframe_box = score_tuples[0][3]

    if ball_keyframe_box is None:
        logger.error("no ball found. quitting with error")
        return None

    logger.info(
        f"selected ball: model={score_tuples[0][0]}, "
        f"score={round(score_tuples[0][2], 3)}, box={ball_keyframe_box}"
    )

    # Clean up after ball detection
    del dino_model, dino_processor, dfine_model, dfine_processor, yolo_model
    gc.collect()
    torch.cuda.empty_cache()

    # Check that the identified ball intersects with the valid shot region. Otherwise, quit
    hoop_keyframe = estimate_hoop_at_timestamp(hoops, ball_keyframe_ts)
    shot_keyframe_box = get_shot_box_from_hoop_box(hoop_keyframe.box, image_size=256)
    logger.info(
        f"keyframe ts {ball_keyframe_ts} - hoop box {hoop_keyframe.box}, shot box {shot_keyframe_box}"
    )

    # Optionally, we should visualize the ball box, hoop box, and shot box
    if visualize:
        start_time = tick()
        plt.close("all")
        out_path = join(shot_dir, "pred.jpg")
        viz_im = Image.open(ball_keyframe_path)
        width, height = viz_im.size
        figsize = (width / 100, height / 100)
        _, ax = plt.subplots(figsize=figsize)
        ax.imshow(viz_im)  # type: ignore
        ax.axis("off")  # type: ignore
        plot_bbox(ball_keyframe_box, ax, "#e56997", label="ball")  # type: ignore
        plot_bbox(hoop_keyframe.box, ax, "#fbc740", label="hoop")  # type: ignore
        plot_bbox(shot_keyframe_box, ax, "#66d2d6", label="shot")  # type: ignore
        plt.savefig(out_path)
        end_time = tick()
        logger.info(
            f"saved box predictions to {out_path} - "
            f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
        )

    if compute_bbox_iou(ball_keyframe_box[:4], shot_keyframe_box[:4]) == 0:
        logger.error(
            "found no intersection between keyframe ball & shot box. quitting with error"
        )
        return None

    # --- Persons detection ---

    # Do person inference on the ball keyframe
    start_time = tick()
    # Important to set the minimum scores to be very low so we can catch outliers
    person_keyframe_idx, person_keyframe_path = select_person_keyframe(
        person_keypoints_model,
        frame_dir,
        frame_names,
        ball_keyframe_idx,
        min_box_score=0.1,
        min_keypoint_score=0.1,
    )
    persons = detect_persons(
        person_keypoints_model, person_keyframe_path, min_box_score=min_person_box_score
    )
    end_time = tick()
    logger.info(
        f"detected {len(persons)} player(s) in keyframe {person_keyframe_idx} - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total",
    )

    # --- Object tracking with SAM2 ---

    # Due to SAM2's intricacies, we need to separate tracking people and balls
    # In the future, we may need to separate tracking for each person but for now that is not necessary
    # In the future, we may also want to track multiple hoops and balls but again not important
    ball_prompts = []
    person_prompts = []
    hoop_prompts = []
    ball_id = 0
    hoop_id = 1
    person_ids = []

    # Annotate hoop (at the keyframe idx) into the prompt
    hoop_prompts = add_hoop_to_tracking_prompt(
        hoop_prompts, ball_keyframe_idx, hoop_id, hoop_keyframe.box
    )

    # Annotate ball into the prompt
    ball_prompts = add_ball_to_tracking_prompt(
        ball_prompts, ball_keyframe_idx, ball_id, ball_keyframe_box, use_box=True
    )

    for j in range(len(persons)):
        # +2 b/c need to account for ball and hoop
        person_prompts = add_person_to_tracking_prompt(
            person_prompts,
            person_keyframe_idx,
            j + 2,
            persons[j],
            min_score=min_person_keypoint_score,
        )
        person_ids.append(j + 2)  # important!

    logger.info(
        f"simulating clicks through {len(ball_prompts)} ball prompts | "
        f"{len(hoop_prompts)} hoop prompts | {len(person_prompts)} person prompts"
    )

    # Do tracking with SAM2
    start_time = tick()
    inference_state = init_inference_state(sam_model, frame_dir)
    video_segments_ball_hoop = track_objects(
        sam_model, ball_prompts + hoop_prompts, inference_state
    )
    end_time = tick()
    logger.info(
        f"finished tracking ball and hoop - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )
    start_time = tick()
    inference_state = init_inference_state(sam_model, frame_dir)
    video_segments_persons = track_objects(sam_model, person_prompts, inference_state)
    end_time = tick()
    logger.info(
        f"finished tracking persons - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )
    start_time = tick()
    video_segments = merge_video_segments(
        video_segments_ball_hoop, video_segments_persons
    )
    end_time = tick()
    logger.info(
        f"merged segments into one - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )

    # Clean up after object tracking
    del sam_model
    gc.collect()
    torch.cuda.empty_cache()

    # Derive bounding boxes from masks once to save compute later
    start_time = tick()
    video_boxes = video_segments_to_video_boxes(video_segments)
    end_time = tick()
    logger.info(
        f"finding bounding boxes from video segments - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )

    # Identify collisions in tracked objects and make a call on which ones to keep
    start_time = tick()
    collisions = label_collisions(video_boxes, ball_id, hoop_id, person_ids)
    for collision_id, collision_ts in collisions:
        logger.info(
            f"found collision for person {collision_id} start at {collision_ts}"
        )
    if len(collisions) > 0:
        video_segments, video_boxes = remove_collisions(
            video_segments, video_boxes, collisions, video_width, video_height
        )
    end_time = tick()
    if len(collisions) == 0:
        logger.info(
            f"no collisions found - "
            f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
        )
    else:
        logger.info(
            f"removed object collisions - "
            f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
        )

    # --- Person keypoint detection ---

    start_time = tick()
    video_skeletons = infer_person_keypoints(
        person_keypoints_model,
        video_boxes,
        person_ids,
        frame_dir,
        frame_names,
        min_match_iou=min_segment_skeleton_iou,
    )
    end_time = tick()
    logger.info(
        f"predicted person keypoints - {end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )

    del person_keypoints_model  # clean up

    # --- Visualize masks on video ---

    if visualize:
        # render the segmentation results every frame
        # This is an expensive operation so use it sparingly
        start_time = tick()
        plt.close("all")
        fourcc = cv2.VideoWriter.fourcc(*"avc1")  # Try to use avc1 codec aka H.264
        out_path = join(shot_dir, "pred.mp4")
        video_writer = cv2.VideoWriter(
            out_path, fourcc, 30, (video_width, video_height)
        )
        if not video_writer.isOpened():  # Try to use XVID codec
            logger.error(
                "> failed to initialize VideoWriter with 'avc1' codec; falling back to 'xvid'"
            )
            fourcc = cv2.VideoWriter.fourcc(*"xvid")
            video_writer = cv2.VideoWriter(
                out_path, fourcc, 30, (video_width, video_height)
            )
        if not video_writer.isOpened():
            raise Exception(
                f"Error: Could not open video file for writing at {out_path}"
            )
        for out_frame_idx in tqdm(range(0, len(frame_names)), desc="writing video"):
            fig, ax = plt.subplots(
                figsize=(video_width / 100, video_height / 100), dpi=100
            )
            ax.set_title(f"frame {out_frame_idx}")  # type: ignore
            ax.imshow(  # type: ignore
                Image.open(join(frame_dir, frame_names[out_frame_idx])).convert("RGB")
            )
            ax.axis("off")  # type: ignore
            if out_frame_idx in video_segments:  # ball may not be in frame
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    show_mask(out_mask, ax, obj_id=out_obj_id)  # type: ignore
            out_frame = fig2img(fig)
            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(out_frame)
            plt.close(fig)
        video_writer.release()
        end_time = tick()
        logger.info(
            f"saved segmentation video to {out_path} - "
            f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
        )

    # --- Classify ball possessions ---

    start_time = tick()
    frame_possessions, _ = classify_possessions(
        possession_model,
        possession_embedder,
        ball_id,
        person_ids,
        frame_idxs,
        frame_paths,
        video_boxes,
        batch_size=32,
    )
    end_time = tick()
    logger.info(
        f"classified possessions to all frames - {end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )

    # Find indices we determine are candidates for holding the ball
    is_holding = assign_possessions(
        ball_id,
        person_ids,
        frame_idxs,
        video_segments,
        video_boxes,
        video_skeletons,
        min_person_keypoint_score=min_person_keypoint_score,
        dist_pixel_thres=dist_pixel_thres,
        max_ball_person_ratio=max_ball_person_ratio,
        fast_mode=True,
    )
    # Only keep those indices as holding
    frame_possessions = [
        frame_possessions[i] if (is_holding[i] is not None) else None
        for i in range(len(frame_possessions))
    ]
    # Find the index the ball was last held by the scorer
    scorer_id, scorer_possession_last_idx = detect_holder(
        ball_keyframe_idx, frame_possessions, window_size=5
    )

    if (scorer_id is None) or (scorer_possession_last_idx is None):
        logger.error(
            f"scorer still not found for clip {start_ts} to {end_ts}. "
            "attempting to progress onwards without a scorer"
        )
        scorer_possession_first_idx = None
    else:
        scorer_possession_first_idx = find_first_possession_from_index(
            scorer_possession_last_idx, frame_possessions
        )
        logger.info(
            f"scorer: {scorer_id} | possession frame idxs {scorer_possession_first_idx} "
            f"to {scorer_possession_last_idx} (subsampled indices)",
        )

    # --- Classify shot outcomes ---

    start_time = tick()
    shot_outcome_by_ball, shot_above_idx, shot_below_idx = detect_shot_outcome(
        frame_names,
        video_height,
        video_width,
        0 if scorer_possession_last_idx is None else scorer_possession_last_idx,
        ball_id,
        hoop_id,
        video_segments,
        video_boxes,
    )
    end_time = tick()
    logger.info(
        f"shot heuristic by ball trajectory: {shot_outcome_by_ball} | "
        f"shot frames {shot_above_idx} to {shot_below_idx} - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )

    # Handle case where we have a pass that intersects the backboard rather than a shot
    # These tend to be below the rim the whole time
    if (shot_above_idx is None) or (shot_below_idx is None):
        logger.warning(
            "decided false positive - could not find shot indices, skipping shot"
        )
        return None

    # Create video for shot classification
    start_time = tick()
    # We first truncate the video to a short area around the shot. We do this with FFMPEG
    # because it is faster, more accurate at timestamps, and also normalizes FPS / quality.
    truncated_path = join(shot_dir, "truncated.mp4")
    try:
        truncate_clip_ffmpeg(
            video_file,
            ball_key_ts - 0.5,
            ball_key_ts + 1,
            truncated_path,
            out_fps=30,
        )
    except Exception as e:
        logger.error(f"Error truncating clip: {e}")
        return None
    # Then use CV2 to perform a per frame crop
    shot_path = join(shot_dir, "shot.mp4")
    create_shot_classification_video(
        truncated_path,
        hoops,
        shot_path,
        offset_ts=ball_key_ts - 0.5,
        image_size=256,
        out_fps=30,
    )
    end_time = tick()
    logger.info(
        f"extracted shot clip ({round(ball_key_ts - 0.5, 2)} to {round(ball_key_ts + 1, 2)}) to {shot_path} - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )

    # Do inference with the model
    device = torch.device(device_name)
    shot_video, _, _ = read_video(shot_path, pts_unit="sec")  # t,h,w,c
    shot_video = shot_video.float().permute(3, 0, 1, 2)  # c,t,h,w

    start_time = tick()
    shot_video = x3d_preprocess_video(shot_video, model_name="x3d_m")
    shot_outcome_probs = x3d_infer_video(shot_model, shot_video.to(device))
    shot_outcome_by_model = shot_outcome_probs > min_shot_score
    end_time = tick()
    logger.info(
        f"shot classification: {shot_outcome_by_model} | prob: {shot_outcome_probs} - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )

    del shot_model, shot_video  # clean up
    gc.collect()
    torch.cuda.empty_cache()

    # We keep the model prediction as the main one over the shot outcome
    shot_outcome = shot_outcome_by_model

    # --- Assign scorer, rebounder, and assister ---

    if scorer_id is not None:
        assert scorer_possession_first_idx is not None, (
            f"scorer_possession_first_idx is None for scorer_id {scorer_id}"
        )
        assert scorer_possession_last_idx is not None, (
            f"scorer_possession_last_idx is None for scorer_id {scorer_id}"
        )
        # We do not know apriori if these are actual assists or rebounds. These are currently candidates.
        # To keep things flexible that will be something we decide live
        start_time = tick()
        assister_id, assister_possession_last_idx = detect_holder(
            scorer_possession_first_idx - 1,
            frame_possessions,
            ignore_ids=[scorer_id],
            window_size=5,
        )
        rebounder_id, rebounder_possession_first_idx = detect_holder(
            shot_below_idx,
            frame_possessions,
            reverse=True,
            window_size=5,
        )
        end_time = tick()

        if assister_id is not None:
            logger.info(
                f"assister {assister_id} | last possession frame {assister_possession_last_idx} - "
                f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
            )
        if rebounder_id is not None:
            logger.info(
                f"rebounder {rebounder_id} | first possession frame {rebounder_possession_first_idx} - "
                f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
            )

        # Find location info for scorer
        scorer_possession_first_bbox = video_boxes[scorer_possession_first_idx][
            scorer_id
        ]
        scorer_possession_last_bbox = video_boxes[scorer_possession_last_idx][scorer_id]

        # Find location info for assister
        assister_possession_last_bbox = None
        if (assister_id is not None) and (assister_possession_last_idx is not None):
            assister_possession_last_bbox = video_boxes[assister_possession_last_idx][
                assister_id
            ]

        # Find location info for rebounder
        rebounder_possession_first_bbox = None
        if (rebounder_id is not None) and (rebounder_possession_first_idx is not None):
            rebounder_possession_first_bbox = video_boxes[
                rebounder_possession_first_idx
            ][rebounder_id]
    else:
        # Default everything to None if we did not find a scorer
        assister_id = None
        rebounder_id = None
        assister_possession_last_idx = None
        rebounder_possession_first_idx = None
        scorer_possession_first_bbox = None
        scorer_possession_last_bbox = None
        assister_possession_last_bbox = None
        rebounder_possession_first_bbox = None

    # --- Compute player embeddings ---

    start_time = tick()
    embeddings: Dict[int, PlayerEmbedding] = {}
    tag_dir = join(shot_dir, "tags")
    makedirs(tag_dir, exist_ok=True)
    for player_id in [scorer_id, assister_id, rebounder_id]:
        if player_id is None:
            continue
        # Compute embeddings for each image and compute k-max pool on top
        image_dir = join(shot_dir, "players", str(player_id))
        player_data = None
        for min_keypoint_present in [10, 5]:
            player_data = embed_player_trajectory(
                embed_model,
                frame_dir,
                frame_names,
                video_segments,
                video_boxes,
                video_skeletons,
                start_frame_idx=0,
                end_frame_idx=len(frame_names) - 1,
                player_id=player_id,
                num_samples=10,
                num_candidates=50,
                min_keypoint_prob=min_person_keypoint_score,
                min_keypoint_present=min_keypoint_present,
                min_bbox_coverage=0.2,
                top_k=3,
                out_dir=image_dir,
            )
            if player_data is not None:
                break
        if player_data is None:
            # Punt if we can't find embeddings for a player we expect to
            continue
        # Copy the featured image to be a tag image for this player
        _, file_extension = splitext(player_data.feat_name)
        tag_name = f"{generate_uuid()}{file_extension}"
        shutil.copyfile(join(image_dir, player_data.feat_name), join(tag_dir, tag_name))
        # Create a tag to save
        tag = TagArtifact(id=player_id, name=tag_name)
        player_data.tag_artifact = tag
        # Set the correct action ID
        if player_id == scorer_id:
            player_data.action_id = ACTION_SHOT
        elif player_id == assister_id:
            player_data.action_id = ACTION_ASSIST
        elif player_id == rebounder_id:
            player_data.action_id = ACTION_REBOUND
        else:
            raise ValueError(
                f"player id {player_id} not found in scorer, assister, or rebounder"
            )
        # Save to the embeddings dict
        embeddings[player_id] = player_data
    end_time = tick()
    logger.info(
        f"computed {len(embeddings)} player embeddings - "
        f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
    )

    scorer_possession_first_ts = (
        frame_ts[scorer_possession_first_idx]
        if scorer_possession_first_idx is not None
        else None
    )
    scorer_possession_last_ts = (
        frame_ts[scorer_possession_last_idx]
        if scorer_possession_last_idx is not None
        else None
    )
    assister_possession_last_ts = (
        frame_ts[assister_possession_last_idx]
        if assister_possession_last_idx is not None
        else None
    )
    rebounder_possession_first_ts = (
        frame_ts[rebounder_possession_first_idx]
        if rebounder_possession_first_idx is not None
        else None
    )

    # If we cannot find images of the assister, then pretend we didn't find one
    # since these are likely low quality
    if (assister_id is not None) and (assister_id not in embeddings):
        assister_id = None
        assister_possession_last_idx = None
        assister_possession_last_ts = None
        assister_possession_last_bbox = None
    # Same for the rebounder
    if (rebounder_id is not None) and (rebounder_id not in embeddings):
        rebounder_id = None
        rebounder_possession_first_idx = None
        rebounder_possession_first_ts = None
        rebounder_possession_first_bbox = None

    # --- Save artifacts ---

    output = {
        "shot_idx": shot_idx,
        "start_ts": start_ts,  # start timestamp for this clip
        "end_ts": end_ts,  # end timestamp for this clip
        "clip_length": end_ts - start_ts,  # clip length in seconds
        "ball_id": ball_id,  # ball identifier
        "hoop_id": hoop_id,  # hoop identifier
        "person_ids": person_ids,  # person / player identifiers
        "keyframe_ball_ts": ball_key_ts,  # timestamp for the ball keyframe
        "keyframe_ball_idx": ball_keyframe_idx,  # frame index for the ball keyframe
        "keyframe_person_ts": frame_ts[
            person_keyframe_idx
        ],  # timestamp for the players keyframe
        "keyframe_person_idx": person_keyframe_idx,  # frame index for the players keyframe
        "scorer_id": scorer_id,  # id for the scorer (must exist)
        "scorer_possession_first_idx": scorer_possession_first_idx,  # first frame index for when the scorer held possession of ball
        "scorer_possession_first_ts": scorer_possession_first_ts,  # first timestamp for when the scorer held possession of ball
        "scorer_possession_first_bbox": scorer_possession_first_bbox,  # bounding box for the scorer at the start of their possession
        "scorer_possession_last_idx": scorer_possession_last_idx,  # last frame index for when the scorer held possession of ball
        "scorer_possession_last_ts": scorer_possession_last_ts,  # last timestamp for when the scorer held possession of ball
        "scorer_possession_last_bbox": scorer_possession_last_bbox,  # bounding box for the scorer at the end of their possession
        "shot_outcome": shot_outcome,  # outcome for the shot
        "shot_outcome_by_ball": shot_outcome_by_ball,  # predicted outcome using a ball heuristic
        "shot_outcome_by_model": shot_outcome_by_model,  # predicted outcome using a trained X3D model
        "shot_above_idx": shot_above_idx,  # last frame index where the ball is above the hoop
        "shot_below_idx": shot_below_idx,  # first frame where the ball is below the hoop
        "shot_above_ts": frame_ts[
            shot_above_idx
        ],  # last frame index where the ball is above the hoop
        "shot_below_ts": frame_ts[
            shot_below_idx
        ],  # first frame where the ball is below the hoop
        "assister_id": assister_id,  # id for the assister (if one exists)
        "assister_possession_last_idx": assister_possession_last_idx,  # last frame index the assister holds onto the ball before passing
        "assister_possession_last_ts": assister_possession_last_ts,  # last timestamp the assister holds onto the ball before passing
        "assister_possession_last_bbox": assister_possession_last_bbox,  # bounding box for the assister when they pass the ball
        "rebounder_id": rebounder_id,  # id for the rebounder (if one exists)
        "rebounder_possession_first_idx": rebounder_possession_first_idx,  # first frame index the rebounder holds onto the ball after the shot
        "rebounder_possession_first_ts": rebounder_possession_first_ts,  # first timestamp the rebounder holds onto the ball after the shot
        "rebounder_possession_first_bbox": rebounder_possession_first_bbox,  # bounding box for the rebounder when they obtain the ball
        "paint_keypoints": None,
    }
    logger.info(output)  # print the result

    # By default these are all empty
    artifacts = ClipArtifacts()

    if save_artifacts:
        # Save clip level outputs
        start_time = tick()
        out_path = join(artifact_dir, "labels.json")
        crop_path = join(artifact_dir, "crops")
        possessions_path = join(artifact_dir, "possessions.json")
        to_json(output, out_path)
        shutil.make_archive(crop_path, "zip", join(shot_dir, "players"))
        to_json(frame_possessions, possessions_path)
        end_time = tick()
        logger.info(
            f"saving outputs - {end_time - start_time}s elapsed - {end_time - fn_start_time} total"
        )

        # Merge the frames back into the main dir
        start_time = tick()
        merge_subsampled_frames_back(frame_dir, all_frame_dir)
        end_time = tick()
        logger.info(
            f"move frames back - {end_time - start_time}s elapsed - {end_time - fn_start_time} total"
        )

        # Create highlight - pass the directory with all frames
        preview_name = "preview.jpg"
        loading_name = "loading.jpg"
        highlight_name = "highlight.mp4"
        start_time = tick()
        create_highlight(
            clip_path,
            video_width,
            video_height,
            all_frame_dir,
            all_frame_names,
            video_segments,
            ball_id,
            artifact_dir,
            logger=logger,
            out_video_name=highlight_name,
            out_preview_name=preview_name,
            out_loading_name=loading_name,
            smoothing_window=10,
            # use last posession index as preview index
            preview_frame_idx=0
            if scorer_possession_last_idx is None
            else scorer_possession_last_idx,
            loading_frame_idx=0,
            sample_rate=frame_sample_rate,
            make_preview=True,
        )
        end_time = tick()
        logger.info(
            f"created highlight - {end_time - start_time}s elapsed - {end_time - fn_start_time} total"
        )

        # Upload artifacts to firebase
        if upload_firebase and (firebase_bucket is not None):
            # Upload all general artifacts
            start_time = tick()
            artifact_names = [
                "trace.log",
                "labels.json",
                "possessions.json",
                "crops.zip",
                preview_name,
                loading_name,
                highlight_name,
            ]
            artifact_blobs = upload_files_to_firebase(
                artifact_names, artifact_dir, firebase_bucket
            )
            artifacts = ClipArtifacts(**artifact_blobs)
            end_time = tick()
            logger.info(
                f"uploaded {len(artifact_names)} artifacts to firebase {firebase_bucket} - "
                f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
            )
            # Upload tag images for players
            start_time = tick()
            tag_bucket = join(firebase_bucket, "tags")
            tag_names = [
                em.tag_artifact.name
                for _, em in embeddings.items()
                if em.tag_artifact is not None
            ]
            tag_blobs = upload_files_to_firebase(tag_names, tag_dir, tag_bucket)
            end_time = tick()
            logger.info(
                f"uploaded {len(tag_names)} tags to firebase {firebase_bucket} - "
                f"{end_time - start_time}s elapsed - {end_time - fn_start_time} total"
            )
            # Set the missing field
            for k in embeddings.keys():
                tag_artifact = embeddings[k].tag_artifact
                if tag_artifact is not None:
                    tag_artifact.url = tag_blobs[tag_artifact.name.replace(".", "_")]
                    embeddings[k].tag_artifact = tag_artifact

    # For shots with missing scorers, this embeddings field will be an empty list
    output = {
        "embeddings": [emb for _, emb in embeddings.items()],
        "artifacts": artifacts,
        **output,
    }
    output = ClipOutput(**output)

    if upload_firebase and (firebase_bucket is not None):
        # Set additional fields in ClipOutput
        if (scorer_id is not None) and (scorer_id in embeddings):
            tag_artifact = embeddings[scorer_id].tag_artifact
            if tag_artifact is not None:
                output.scorer_image_url = tag_artifact.url
        if (assister_id is not None) and (assister_id in embeddings):
            tag_artifact = embeddings[assister_id].tag_artifact
            if tag_artifact is not None:
                output.assister_image_url = tag_artifact.url
        if (rebounder_id is not None) and (rebounder_id in embeddings):
            tag_artifact = embeddings[rebounder_id].tag_artifact
            if tag_artifact is not None:
                output.rebounder_image_url = tag_artifact.url

    fn_end_time = tick()
    logger.info(f"processing complete - total {fn_end_time - fn_start_time}s elapsed")

    return output


def create_keyframes(
    video_file: str,
    out_dir: str,
    paused_ts: Union[List[float], List[int]] = [],
    upload_firebase: bool = False,
    firebase_bucket: Optional[str] = None,
) -> List[KeyframeArtifact]:
    """Sample keyframes from video and upload to firebase."""
    # in practice, we would use the session id
    artifact_dir = join(out_dir, "artifacts")
    makedirs(artifact_dir, exist_ok=True)

    # Create logger for video inference
    log_path = join(artifact_dir, "trace.log")
    if isfile(log_path):
        remove(log_path)
    logger = create_logger("keyframe_logger", log_path)

    # Get video metadata
    video_width, video_height = get_video_frame_size(video_file)
    video_length = get_video_length(video_file)
    logger.info(
        f"found video (length={round(video_length, 2)}s,"
        f"width={video_width},height={video_height})"
    )

    start_time = tick()
    keyframe_dir = join(artifact_dir, "keyframes")
    makedirs(keyframe_dir, exist_ok=True)
    keyframe_names = []

    # Get keyframe timestamps
    keyframe_ts = get_keyframe_timestamps(
        paused_ts, video_length, min_diff_ts=5, nearby_ts=1
    )
    logger.info(f"creating {len(keyframe_ts)} keyframes at {keyframe_ts}")
    for i, row in enumerate(keyframe_ts):
        keyframe_name = f"{i:04d}.jpg"
        keyframe_path = join(keyframe_dir, keyframe_name)
        sample_frame_at_timestamp(video_file, row["ts"], keyframe_path)
        keyframe_names.append(keyframe_name)
        keyframe_ts[i]["name"] = keyframe_name  # type: ignore
        keyframe_ts[i]["width"] = video_width
        keyframe_ts[i]["height"] = video_height
    end_time = tick()
    logger.info(
        f"done. created {len(keyframe_ts)} keyframes - {end_time - start_time}s elapsed"
    )

    keyframe_artifacts: List[KeyframeArtifact] = []
    if upload_firebase and (firebase_bucket is not None):
        # Upload keyframes to firebase
        start_time = tick()
        keyframe_urls = upload_files_to_firebase(
            keyframe_names, keyframe_dir, join(firebase_bucket, "keyframes")
        )
        for row in keyframe_ts:
            artifact = KeyframeArtifact(
                idx=int(row["idx"]),
                ts=row["ts"],
                start_ts=row["start_ts"],
                end_ts=row["end_ts"],
                width=int(row["width"]),
                height=int(row["height"]),
                url=keyframe_urls[str(row["name"]).replace(".", "_")],
            )
            keyframe_artifacts.append(artifact)
        logger.info(
            f"uploaded {len(keyframe_names)} keyframes to storage {firebase_bucket} - {end_time - start_time}s elapsed"
        )

    return keyframe_artifacts
