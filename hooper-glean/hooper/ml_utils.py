import cv2
import torch
import logging
import numpy as np
import numpy.typing as npt
from statistics import mode, StatisticsError
from tqdm import tqdm
from scipy import stats
from PIL import Image
from os import makedirs
from os.path import join
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple, Any, Optional, Set
from transformers import CLIPVisionModel
from shared_utils.types import (
    PromptType,
    BoxType,
    SAM2SegmentsType,
    SAM2BoxesType,
    SAM2SkeletonsType,
    PlayerEmbedding,
    HoopOutput,
    ClusterOutput,
    TagArtifact
)
from .solider_utils import SoliderFeatureExtractor
from .utils import (
    load_image,
    build_positive_clicks_from_ball,
    build_positive_clicks_from_skeleton,
    compute_mask_intersection,
    compute_bbox_intersection,
    compute_bbox_iou,
    compute_bbox_area,
    filter_consecutive_numbers,
    interpolate_consecutive_numbers,
    bounding_box_to_binary_mask,
    get_box_center,
    get_video_frame_size,
    is_ball_above_hoop,
    is_ball_below_backboard,
    is_ball_outside_backboard,
    did_ball_go_through_hoop,
    hallucinate_backboard_from_hoop,
    k_max_pool,
    is_ball_near_hand,
    is_ball_near_hand_fast,
    compute_dist_ball_to_hands,
    compute_dist_ball_to_hands_fast,
    find_hoop_neighbor_idxs_to_timestamp,
    interpolate_hoops,
)
from .shot_utils import (
    detect_shot,
    interpolate_shots,
    find_shot_intervals,
    merge_nearby_intervals,
)
from .possession_utils import infer_possessor_batch, get_possession_model_transforms
from .possession_utils import HolderFrameClassifier
from .utils import ACTION_SHOT



def add_hoop_to_tracking_prompt(
    prompts: List[PromptType],
    frame_idx: int,
    obj_id: int,
    box: BoxType,
) -> List[PromptType]:
    r"""Annotate the hoop for tracking using a bounding box.
    :note: the goal of this is to handle unexpected movements in the hoop during the clip
    :param prompts:
    :param frame_idx: Frame for the annotation
    :param obj_id: Object id to track
    :param box: Bounding box for hoop
    """
    # box is expected to be xyxy coordinates
    # (obj_id, frame_idx, points, labels, box)
    prompts.append((obj_id, frame_idx, None, None, box))
    return prompts


def add_ball_to_tracking_prompt(
    prompts: List[PromptType],
    frame_idx: int,
    obj_id: int,
    box: BoxType,
    use_box: bool = False,
) -> List[PromptType]:
    r"""Add a ball box to a prompt of annotated keypoints for SAM2.
    :param prompts: Current prompt
    :param frame_idx: Index of frame for this ball bounding box
    :param obj_id: Identifier for object
    :param box: [x1, y1, x2, y2, score]
    :param use_box: Whether to use the box or not
    :return prompts: List of annotated objects
    """
    if use_box:
        prompts.append((obj_id, frame_idx, None, None, box[:4]))
    else:
        # Build prompt from ball
        points, labels = build_positive_clicks_from_ball(box)
        if len(points) == 0 or len(labels) == 0:
            raise Exception("Ball produced invalid points and labels for SAM2")
        # (obj_id, frame_idx, points, labels, box)
        prompts.append((obj_id, frame_idx, points, labels, None))
    return prompts


def add_person_to_tracking_prompt(
    prompts: List[PromptType],
    frame_idx: int,
    obj_id: int,
    keypoints: npt.NDArray,
    min_score: float = 0.15,
) -> List[PromptType]:
    r"""Add a person skeleton to a prompt of annotated keypoints for SAM2.
    :param prompts: Current prompt
    :param frame_idx: Index of frame for this ball bounding box
    :param obj_id: Identifier for object
    :param keypoints
    :param min_score: Minimum score to keep a a keypoint
    :return prompts: Dictionary of annotated objects
    """
    points, labels = build_positive_clicks_from_skeleton(keypoints, min_score=min_score)
    if len(points) == 0 or len(labels) == 0:
        return prompts
    # (obj_id, frame_idx, points, labels, box)
    prompts.append((obj_id, frame_idx, points, labels, None))
    return prompts


def init_inference_state(sam_model: Any, video_dir: str) -> Any:
    """Load an initial inference state for SAM2. This is important to not reload frames
    for each call that we want to do.
    :param sam_model: SAM2 model instance
    :param video_dir: Directory of sorted frames
    :return inference_state: Inference state
    """
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = sam_model.init_state(
            video_path=video_dir,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
    return inference_state


def track_objects(
    sam_model: Any,
    prompts: List[PromptType],
    inference_state: Any,
) -> SAM2SegmentsType:
    r"""Use SAM2 to do a forward and backward pass to track multiple objects.
    :param sam_model: SAM2 model instance
    :param prompts: Annotated keypoints for prompts
    :param video_dir: Directory of sorted frames
    :return video_segments: Propagated video segments forward and backward
    """
    if len(prompts) == 0:
        return {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_model._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        sam_model.reset_state(inference_state)

        for obj_id, frame_idx, points, labels, box in prompts:
            # `add_new_points` returns masks for all objects added so far on this interacted frame
            _, out_obj_ids, out_mask_logits = sam_model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                box=box,
            )
        # run propagation throughout the video and collect the results in a dict
        # video_segments contains the per-frame segmentation results
        video_segments = {}
        # run reverse
        for out_frame_idx, out_obj_ids, out_mask_logits in sam_model.propagate_in_video(
            inference_state, reverse=True
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[j] > 0.0).cpu().numpy()
                for j, out_obj_id in enumerate(out_obj_ids)
            }
        # run forward
        for out_frame_idx, out_obj_ids, out_mask_logits in sam_model.propagate_in_video(
            inference_state, reverse=False
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[j] > 0.0).cpu().numpy()
                for j, out_obj_id in enumerate(out_obj_ids)
            }

    return video_segments


def smooth_detected_hoops(
    hoops: List[HoopOutput],
    iou_threshold: float = 0.8,
) -> Tuple[List[HoopOutput], int]:
    """Hoop detections have some natural noise that pose challenges for later operations
    such as frame differencing. We will apply an opinionated smoothing algorithm to ignore
    all minor movements.
    @return smoothed (smoothed hoops)
    @return num_frames_with_movement (number of large movements detected)
    """
    num_frames_with_movement: int = 0
    smoothed: List[HoopOutput] = []
    for i in range(len(hoops)):
        ts_i = hoops[i].ts
        if i == 0:
            smoothed.append(hoops[i])
        else:
            prev_hoop = smoothed[-1].copy()
            # If there is sufficient difference between the previous and current hoop
            # then we save the new hoop, otherwise we keep the previous one
            if compute_bbox_iou(prev_hoop.box, hoops[i].box) < iou_threshold:
                smoothed.append(hoops[i])
                # panning will be quite disruptive and count a lot towards this
                num_frames_with_movement += 1
            else:
                # clone prev hoop but with a new timestamp
                smoothed.append(
                    HoopOutput(ts=ts_i, box=prev_hoop.box, prob=prev_hoop.prob)
                )
    return smoothed, num_frames_with_movement


def score_ball_detections(
    balls: List[BoxType],
    hoop: BoxType,
    image_height: int,
    apex_weight: float = 0.5,
    conf_weight: float = 0.5,
) -> List[float]:
    """Find the key candidate ball as ball at the apex but still in the shot region.
    Weight the score between the apex and the probability returned by the model.
    """
    if len(balls) == 0:
        return []  # nothing to do
    # Compute the shot region from the hoop box
    shot_box = get_shot_box_from_hoop_box(hoop, image_size=256)
    scores = []
    for i in range(len(balls)):
        ball_box = balls[i][:4]
        conf_score = balls[i][4]
        # If the ball is not in the shot region, skip it
        if compute_bbox_iou(ball_box, shot_box) == 0:
            score = 0
        else:
            x1, y1, x2, y2 = ball_box
            ball_height = (y1 + y2) / 2.0
            # if the ball is at the top of the frame => ball_height ~ 0, apex_score ~ 1
            # if the ball is at the bottom of the frame => ball_height ~ image_height, apex_score ~ 0
            apex_score = 1 - (ball_height / image_height)
            score = apex_weight * apex_score + conf_weight * conf_score
        scores.append(score)
    return scores


def score_ensemble_detections(
    balls: List[BoxType],
    others: List[List[BoxType]],
    min_iou: float = 0.5,
) -> List[float]:
    """Given a set of model detections, score them compared with other models.
    @param balls: List[BoxType] - List of ball bounding boxes
    @param others: List[List[BoxType]] - List of lists of other model detections
    @param min_iou: float - Minimum intersection over union threshold for agreement
    @return scores: List[float] - List of agreement ensemble scores
    """
    scores = []
    for i in range(len(balls)):
        ball = balls[i]
        # Compute agreement scores
        agreements: List[bool] = []
        for j in range(len(others)):
            agreement_j = False
            for k in range(len(others[j])):
                candidate = others[j][k]
                if compute_bbox_iou(ball, candidate) >= min_iou:
                    agreement_j = True
                    break
            agreements.append(agreement_j)
        score = sum(agreements) + 1  # min score is 1
        scores.append(score)
    return scores


@torch.no_grad()
def select_person_keyframe(
    model,
    frame_dir: str,
    frame_names: List[str],
    ball_keyframe_idx: int,
    min_box_score: float = 0.1,
    min_keypoint_score: float = 0.1,
) -> Tuple[int, str]:
    r"""Pick the right person keyframe to annotate.
    :note: we could not get away with using YOLO because it could not handle small people well
           instead, we must pay the tax of using Detectron2, which is fine since this is a finite set.
           The difference is night and day.
    :note: we set the minimum scores low because this is a search problem not a detection problem. We would rather err
           on the side of finding more than filtering too much.
    :param frame_dir: Directory for frame images
    :param frame_names: List of files for different frames
    :param min_box_score: Minimum confidence score for a person bounding box (default: 0.1)
    :param min_keypoint_score: Minimum confidence score for a person keypoint (default: 0.1)
    :return person_keyframe_idx:
    :return person_keyframe_path:
    """
    candidate_paths = []
    candidate_idxs = []
    candidate_ims = []

    # Load the main candidate image
    candidate_path_cur = join(frame_dir, frame_names[ball_keyframe_idx])
    candidate_im_cur = load_image(candidate_path_cur)
    candidate_paths.append(candidate_path_cur)
    candidate_idxs.append(ball_keyframe_idx)
    candidate_ims.append(candidate_im_cur)

    # Load a set of sampled surrounding images
    # for delta_idx in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
    for delta_idx in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
        candidate_idx_pre = ball_keyframe_idx - delta_idx
        candidate_idx_post = ball_keyframe_idx + delta_idx

        if candidate_idx_pre > 0:
            candidate_path_pre = join(frame_dir, frame_names[candidate_idx_pre])
            candidate_im_pre = load_image(candidate_path_pre)
            candidate_paths.append(candidate_path_pre)
            candidate_idxs.append(candidate_idx_pre)
            candidate_ims.append(candidate_im_pre)
        if candidate_idx_post < len(frame_names):
            candidate_path_post = join(frame_dir, frame_names[candidate_idx_post])
            candidate_im_post = load_image(candidate_path_post)
            candidate_paths.append(candidate_path_post)
            candidate_idxs.append(candidate_idx_post)
            candidate_ims.append(candidate_im_post)

    # Pass to detectron model for inference
    candidate_batch = np.stack(candidate_ims)
    results = model(candidate_batch)

    box_scores = []
    keypoint_scores = []
    for i in range(len(results)):
        instances = results[i]["instances"]

        valid_idxs = []
        if len(instances.pred_boxes) == 0:
            box_score = 0  # default to 0 if no boxes
        else:
            # score 1 - people with bounding boxes of a minimum confidence level
            box_confs = instances.scores.cpu().numpy().tolist()  # |person|
            box_areas = [
                int(compute_bbox_area(box))
                for box in instances.pred_boxes.tensor.cpu().numpy().tolist()
            ]
            area_cutoff = int(max(box_areas) / 8)
            valid_idxs = [
                i
                for i, c in enumerate(box_confs)
                if ((c > min_box_score) and (box_areas[i] > area_cutoff))
            ]
            box_score = sum([box_confs[idx] for idx in valid_idxs])

        if len(instances.pred_keypoints) == 0 or len(valid_idxs) == 0:
            keypoint_score = 0  # default to 0 if no keypoints
        else:
            # score 2 - people with number of keypoints above a confidence level
            keypoint_confs = instances.pred_keypoints.cpu().numpy()[
                :, :, -1
            ]  # |person| x |keypoints|
            # only keep keypoints where the bounding box is kept
            keypoint_confs = keypoint_confs[valid_idxs]
            keypoint_counts = np.sum(
                keypoint_confs >= min_keypoint_score, axis=1
            )  # |person|
            keypoint_score = sum(keypoint_counts.tolist())

        box_scores.append(box_score)
        keypoint_scores.append(keypoint_score)

    indices = list(range(len(results)))
    elements = list(zip(indices, keypoint_scores, box_scores))
    elements = sorted(elements, key=lambda x: (x[1], x[2]))
    best_idx = int(elements[-1][0])

    return candidate_idxs[best_idx], candidate_paths[best_idx]


def detect_persons(
    model, image_path: str, min_box_score: float = 0.1
) -> List[npt.NDArray]:
    r"""Detect person skeletons from the image frame.
    :param model: Detectron2 model
    :param image_path: Path to an image frame
    :return keypoints: List of keypoints
    """
    im_bgr = load_image(image_path)
    preds = model([im_bgr])[0]
    instances = preds["instances"]
    num_persons_raw = len(instances)
    # Heuristic - filter out people by box sizes!
    box_confs = instances.scores.cpu().numpy().tolist()
    box_areas = [
        int(compute_bbox_area(box))
        for box in instances.pred_boxes.tensor.cpu().numpy().tolist()
    ]
    area_cutoff = int(max(box_areas) / 8)
    valid_idxs = [
        i
        for i, c in enumerate(box_confs)
        if ((c > min_box_score) and (box_areas[i] > area_cutoff))
    ]
    num_persons_valid = len(valid_idxs)
    # Fetch the predicted keypoints
    keypoints = instances.pred_keypoints.cpu().numpy()
    # Heuristic - to avoid filtering out important players that are just farther
    # away, we only filter out people if its a significant number of smaller boxes
    # aka an audience
    if num_persons_raw - num_persons_valid > 5:
        keypoints = keypoints[valid_idxs]
    return keypoints


def filter_balls_similar_to_backboard_or_hoop(
    ball_candidates: List[BoxType],
    hoop: BoxType,
    backboard: BoxType,
    max_iou: float = 0.25,
) -> List[BoxType]:
    """Remove any candidates too similar in size and location to the backboard.
    We do this BEFORE the ball selection code.
    """
    filtered = []
    for i in range(len(ball_candidates)):
        ball = ball_candidates[i]
        # We do not filter by the width of the hoop because of perspective
        # A ball near the start of its trajectory may appear larger than the hoop
        if (
            compute_bbox_iou(ball, backboard) < max_iou
            and compute_bbox_iou(ball, hoop) < max_iou
        ):
            filtered.append(ball)
    return filtered


def filter_balls_for_stationary_objects(
    ball_candidates: List[BoxType],
    stationary_objects: List[BoxType],
    min_iou: float = 0.5,
) -> List[BoxType]:
    """Remove any candidates that appear in the same location as a stationary object.
    :param ball_candidates: List of bounding boxes for candidate balls.
    :param stationary_objects: List of bounding boxes for stationary objects.
    :param min_iou: Minimum intersection over union threshold.
    :return: List of filtered bounding boxes.
    """
    filtered = []
    for i in range(len(ball_candidates)):
        is_bad = False
        ball = ball_candidates[i]
        for j in range(len(stationary_objects)):
            obj = stationary_objects[j]
            if compute_bbox_iou(ball, obj) >= min_iou:
                is_bad = True
                break
        if not is_bad:
            filtered.append(ball)
    return filtered


def filter_detected_balls_pipeline(
    ts_batch: List[float],
    balls_batch: List[List[BoxType]],
    hoops_batch: List[HoopOutput],
    video_height: int,
    video_width: int,
) -> List[List[BoxType]]:
    """Filter the raw detections using heuristics.
    Two main filtering techniques:
    1. Filter balls that are too close to hoops.
    2. Filter balls that are stationary objects.
    @param ts_batch: Batch of timestamps
    @param balls_batch: Batch of detected balls per frame
    @param hoops_batch: Detected hoop positions
    @param video_height: Height of the video frame
    @param video_width: Width of the video frame
    """
    assert len(ts_batch) == len(balls_batch) == len(hoops_batch)
    batch_size = len(balls_batch)

    # For each frame, filter out any balls that are too close to hoops / backboards
    filtered_batch_1 = []
    for i in range(batch_size):
        # Use a strict definition to be exact about the backboard location
        backboard_box = hallucinate_backboard_from_hoop(
            hoops_batch[i].box, video_height, video_width
        )
        filtered_i = filter_balls_similar_to_backboard_or_hoop(
            balls_batch[i], hoops_batch[i].box, backboard_box
        )
        filtered_batch_1.append(filtered_i)

    # Across the batch, we should filter for stationary objects by checking all other frames
    filtered_batch_2 = []
    for i in range(batch_size):
        # Pick stationary objects as everything except this index
        stationary_objects = []
        for j in range(batch_size):
            if j != i:
                stationary_objects.extend(filtered_batch_1[j])
        # Filter any objects that stayed in the same place
        filtered_i = filter_balls_for_stationary_objects(
            filtered_batch_1[i],
            stationary_objects,
            min_iou=0.5,
        )
        filtered_batch_2.append(filtered_i)

    return filtered_batch_2


def estimate_hoop_at_timestamp(hoops: List[HoopOutput], query_ts: float) -> HoopOutput:
    """Estimate the hoop bounding box at a given timestamp.
    @param hoops: List of detected hoop positions at certain timestamps
    @param query_ts: Timestamp to estimate the hoop position at
    """
    idx1, idx2 = find_hoop_neighbor_idxs_to_timestamp(hoops, query_ts)
    return interpolate_hoops(hoops[idx1], hoops[idx2], query_ts)


def detect_shot_timestamps(
    video_file: str,
    video_length: int | float,
    hoops: List[HoopOutput],
    iou_threshold: float = 0.8,
    box_padding_bottom: int = 20,
    history_length: int = 10,
    min_detect_shot_score: float = 0.5,
    min_contiguous_length: int = 5,
) -> List[float]:
    """Detect shot timestamps from video file.
    @note: takes into account paused timestamps
    @param video_file: Path to video .mp4 file.
    @param video_length: Length of the video
    @param hoops: List of detected hoop positions at certain timestamps
    @param iou_threshold: Used to reset hoops
    @param min_detect_shot_score: Minimum score to detect a shot
    @param min_contiguous_length: Minimum length of continuous shot discovery to make a interval
    @return key_timestamps: Timestamps of detected shots above the rim
    """
    # Compute delta between images
    reader = cv2.VideoCapture(video_file)
    video_width, video_height = get_video_frame_size(video_file)

    # Use this to track motion
    # Do not use `createBackgroundSubtractorMOG2`
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()  # type: ignore

    frame_idx: int = 0
    outputs: List[Dict] = []
    history: List[npt.NDArray] = []
    last_hoop = None

    pbar = tqdm(total=int(video_length), desc="detecting shots")
    while True:
        success, im_bgr_i = reader.read()
        if not success:
            break
        # Get current timestamp in the reader
        timestamp_i = round(max(0, reader.get(cv2.CAP_PROP_POS_MSEC) / 1000.0), 3)
        # Find the hoop
        hoop = estimate_hoop_at_timestamp(hoops, timestamp_i)
        # Check if the hoop has changed a lot
        if (last_hoop is not None) and (
            compute_bbox_iou(last_hoop.box, hoop.box) < iou_threshold
        ):
            # Reset the background subtractor & history
            fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()  # type: ignore
            print("detected movement in hoop. resetting background subtractor")
        # Estimate backboard to use for shot region of detection
        backboard_box = hallucinate_backboard_from_hoop(
            hoop.box, video_height, video_width
        )
        # Build bounding boxes
        bbox_above = get_upper_shot_region(backboard_box, hoop.box, box_padding_bottom)
        # Compute the hoop width as a parameter
        hoop_width = int(hoop.box[2] - hoop.box[0])
        # Compute the mask
        x1, y1, x2, y2 = bbox_above
        roi = im_bgr_i[y1:y2, x1:x2]
        fgmask = fgbg.apply(roi)
        # cv2.imwrite(f'/Users/mikewu/Downloads/tmp/roi/{timestamp_i}.jpg', roi)
        # cv2.imwrite(f'/Users/mikewu/Downloads/tmp/mask/{timestamp_i}.jpg', fgmask)
        # Update mask history
        history.append(fgmask)
        if len(history) > history_length:
            history.pop(0)
        output, _ = detect_shot(
            fgmask,
            timestamp_i,
            previous_masks=history[:-1],
            ball_radius=hoop_width // 3,
            min_score=min_detect_shot_score,
            trajectory_frames=history_length,
        )
        outputs.append(output)
        frame_idx += 1
        last_hoop = hoop  # Update the next hoop
        pbar.update(int(timestamp_i) - pbar.n)
    pbar.close()
    # Find segments from the raw outputs
    outputs = interpolate_shots(outputs, gap=3)
    # Convert outputs to intervals
    intervals = find_shot_intervals(
        outputs, min_contiguous_length=min_contiguous_length
    )
    intervals = merge_nearby_intervals(intervals, max_gap=1.0)
    # Take the start timestamp for each segment
    key_timestamps = [interval[1] for interval in intervals]
    # Round the timestamps
    key_timestamps = [round(ts, 3) for ts in key_timestamps]
    return key_timestamps


def get_upper_shot_region(
    backboard: BoxType, hoop: BoxType, box_padding_bottom: int = 0
) -> BoxType:
    """Get a region to scan for motion above the rim.
    @param backboard: Bounding box- xyxy, and score probability
    @param hoop: Bounding box- xyxy, and score probability
    @return: Bounding box
    """
    # We take the top of the backboard as the top but the
    # top of the rim as the bottom
    x1, _, x2, _ = backboard[:4]
    _, y2, _, _ = hoop[:4]
    # Cast to integers
    return [int(x1), 0, int(x2), int(y2 + box_padding_bottom)]


def assign_possessions(
    ball_id: int,
    person_ids: List[int],
    frame_idxs: List[int],
    video_segments: SAM2SegmentsType,
    video_boxes: SAM2BoxesType,
    video_skeletons: SAM2SkeletonsType,
    filter_thres: int = 3,
    interp_thres: int = 5,
    min_person_keypoint_score: float = 0.7,
    dist_pixel_thres: int = 10,
    max_ball_person_ratio: int = 30,
    require_box_overlap: bool = True,
    fast_mode: bool = True,
) -> List[Optional[int]]:
    """Build an array of possessions for each person ID.
    @param ball_id: Key ball id
    @param person_ids: Id for each person
    @param frame_idxs: Frame indices
    @param frame_ts: Map from index to frame timestamp
    @param video_segments: Map from frame index to mapping between person IDs and masks
    @param video_boxes: Map from frame index to mapping between person IDs and boxes
    @param video_skeletons: Map from frame index to mapping between person IDs and skeletons
    @param filter_thres: Minimum number of consecutive frames with possession to not filter (default: 3)
    @param interp_thres: Maximum number of consecutive frames with possession to not interp (default: 3)
    @param min_person_keypoint_score: Minimum probability to make a skeleton keypoint valid (default: 0.15)
    @return: Possessions for each frame
    """
    possessions: List[Optional[int]] = []
    # Do a first pass to greedily assign ball possession
    for frame_idx in frame_idxs:
        segments = video_segments[frame_idx]
        boxes = video_boxes.get(frame_idx, None)
        skeletons = video_skeletons.get(frame_idx, {})

        # If no boxes then skip
        if boxes is None:
            possessions.append(None)
            continue

        # If no ball found, then skip
        if ball_id not in segments:
            possessions.append(None)
            continue

        assert ball_id in boxes, f"Ball ID {ball_id} not in boxes but in segments?"

        # Pull the ball mask
        ball_mask = segments[ball_id][0]
        ball_bbox = boxes[ball_id]
        ball_area = 0 if ball_bbox is None else compute_bbox_area(ball_bbox)
        ball_exists = ball_bbox is not None  # check if ball exists

        # Track the checks passed by each user
        person_scores: List[int | float] = []
        holder: Optional[int] = None

        if ball_exists:  # nothing to do if we do not have a ball in the frame
            for person_id in person_ids:
                person_score = 0
                if person_id in segments:
                    # Pull the person mask & skeleton
                    person_mask = segments[person_id][0]
                    person_bbox = boxes[person_id]
                    if person_bbox is not None:
                        person_area = compute_bbox_area(person_bbox)
                        if require_box_overlap:
                            if fast_mode:
                                # +1 if the ball and person bboxes intersect
                                check = (
                                    compute_bbox_intersection(ball_bbox, person_bbox)
                                    > 0
                                )
                            else:
                                # +1 if the ball and person masks intersect (or if we ignore this check)
                                check = (
                                    compute_mask_intersection(ball_mask, person_mask)
                                    > 0
                                )
                            person_score += int(check)
                        # +1 if the ball is near the hand
                        if (person_id in skeletons) and ball_exists:
                            person_skeleton = skeletons[person_id]
                            if fast_mode:
                                check = is_ball_near_hand_fast(
                                    ball_bbox,
                                    person_skeleton,
                                    dist_pixel_thres=dist_pixel_thres,
                                    min_conf=min_person_keypoint_score,
                                )
                            else:
                                check = is_ball_near_hand(
                                    ball_mask,
                                    person_skeleton,
                                    dist_pixel_thres=dist_pixel_thres,
                                    min_conf=min_person_keypoint_score,
                                )
                            person_score += int(check)
                            if check:
                                # +actual distance to the score to break ties
                                if fast_mode:
                                    lh_dist, rh_dist = compute_dist_ball_to_hands_fast(
                                        ball_bbox,
                                        person_skeleton,
                                        min_conf=min_person_keypoint_score,
                                    )
                                else:
                                    lh_dist, rh_dist = compute_dist_ball_to_hands(
                                        ball_mask,
                                        person_skeleton,
                                        min_conf=min_person_keypoint_score,
                                    )
                                min_dist = min(lh_dist, rh_dist)
                                person_score += (
                                    dist_pixel_thres - min_dist
                                )  # the closer it is, the higher the added score
                        # if the ball is too big, set score to 0
                        if (
                            person_area > 0
                            and (ball_area * 100 / person_area) > max_ball_person_ratio
                        ):
                            person_score = 0
                person_scores.append(person_score)

            if max(person_scores) > 0:
                # Assign holder to the be the index with the max score
                holder_idx = int(np.argmax(person_scores))
                holder = person_ids[holder_idx]

        possessions.append(holder)

    # Interpolate possessions
    # Do this before filtering: example - x, x, None, None, x, x
    possessions = interpolate_consecutive_numbers(possessions, thres=interp_thres)
    # Remove short possessions
    possessions = filter_consecutive_numbers(possessions, thres=filter_thres)
    # Interpolate again
    possessions = interpolate_consecutive_numbers(possessions, thres=interp_thres)
    return possessions


def classify_possessions(
    classifier: "HolderFrameClassifier",
    embedder: "CLIPVisionModel",
    ball_id: int,
    person_ids: List[int],
    frame_idxs: List[int],
    image_paths: List[str],
    video_boxes: SAM2BoxesType,
    batch_size: int = 32,
    filter_thres: int = 3,
    interp_thres: int = 5,
    conf_thres: float = 0.5,
) -> Tuple[List[Optional[int]], List[Optional[float]]]:
    """Classify possessions frame by frame.
    @param ball_id: Key ball id
    @param person_ids: Id for each person
    @param frame_idxs: Frame indices
    @param image_paths: Paths to images
    @param video_boxes: Boxes for each person in each frame
    @param batch_size: Batch size for classification
    @param filter_thres: Minimum number of consecutive frames with possession to not filter (default: 3)
    @param interp_thres: Maximum number of consecutive frames with possession to not interp (default: 3)
    """
    assert len(image_paths) == len(frame_idxs)
    assert len(image_paths) == len(video_boxes)

    # Transform for images
    img_transform = get_possession_model_transforms()
    max_candidates: int = classifier.max_candidates or 20

    # Setup output structures
    possessions: List[Optional[int]] = []
    confidences: List[Optional[float]] = []

    # These are to hold onto batch inputs
    batch_img_paths: List[str] = []
    batch_box_ball: List[Optional[BoxType]] = []
    batch_box_candidates: List[List[BoxType]] = []
    batch_candidate_ids: List[List[int]] = []
    batch_possessions: List[Optional[int]] = []
    batch_confidences: List[Optional[float]] = []

    pbar = tqdm(total=len(frame_idxs), desc="classifying possessions")
    for i, frame_idx in enumerate(frame_idxs):
        img_path = image_paths[i]
        boxes = video_boxes.get(frame_idx, None)
        box_ball: Optional[BoxType] = None
        box_candidates: List[BoxType] = []
        candidate_ids: List[int] = []
        if boxes is not None:
            box_ball = boxes[ball_id]
            for person_id in person_ids:
                if person_id in boxes:
                    box_person = boxes[person_id]
                    if box_person is not None:
                        box_candidates.append(box_person)
                        candidate_ids.append(person_id)
        batch_img_paths.append(img_path)
        batch_box_ball.append(box_ball)
        batch_box_candidates.append(box_candidates)
        batch_candidate_ids.append(candidate_ids)

        if len(batch_img_paths) % batch_size == 0:
            # Do inference
            has_holders, has_holder_probs, holder_idxs = infer_possessor_batch(
                classifier,
                embedder,
                img_transform,
                batch_img_paths,
                batch_box_candidates,
                batch_box_ball,
                max_candidates=max_candidates,
                conf_threshold=conf_thres,
            )
            assert len(has_holders) == len(has_holder_probs) == len(holder_idxs)
            for j in range(len(has_holders)):
                has_holder = has_holders[j]
                has_holder_prob = has_holder_probs[j]
                holder_idx = holder_idxs[j]
                person_ids = batch_candidate_ids[j]
                if has_holder and (holder_idx is not None):
                    batch_possessions.append(person_ids[holder_idx])
                else:
                    batch_possessions.append(None)
                batch_confidences.append(has_holder_prob)

            possessions += batch_possessions
            confidences += batch_confidences
            pbar.update(batch_size)

            # Reset the batch entities
            batch_img_paths = []
            batch_box_ball = []
            batch_box_candidates = []
            batch_candidate_ids = []
            batch_possessions = []
            batch_confidences = []

    if len(batch_img_paths) > 0:
        # Do inference
        has_holders, has_holder_probs, holder_idxs = infer_possessor_batch(
            classifier,
            embedder,
            img_transform,
            batch_img_paths,
            batch_box_candidates,
            batch_box_ball,
            max_candidates=max_candidates,
            conf_threshold=conf_thres,
        )
        assert len(has_holders) == len(has_holder_probs) == len(holder_idxs)
        for j in range(len(has_holders)):
            has_holder = has_holders[j]
            has_holder_prob = has_holder_probs[j]
            holder_idx = holder_idxs[j]
            person_ids = batch_candidate_ids[j]
            if has_holder and (holder_idx is not None):
                batch_possessions.append(person_ids[holder_idx])
            else:
                batch_possessions.append(None)
            batch_confidences.append(has_holder_prob)

        possessions += batch_possessions
        confidences += batch_confidences
        pbar.update(len(batch_img_paths))

    pbar.close()

    # Interpolate possessions
    # Do this before filtering: example - x, x, None, None, x, x
    possessions = interpolate_consecutive_numbers(possessions, thres=interp_thres)
    # Remove short possessions
    possessions = filter_consecutive_numbers(possessions, thres=filter_thres)
    # Interpolate again
    possessions = interpolate_consecutive_numbers(possessions, thres=interp_thres)

    return possessions, confidences


def detect_holder(
    frame_idx: int,
    frame_possessions: List[Optional[int]],
    reverse: bool = False,
    ignore_ids: List[int] = [],
    window_size: int = 5,
) -> Tuple[Optional[int], Optional[int]]:
    """Find the ID of the player last/next holding the ball.
    @param frame_idx: Index to search before
    @param frame_possessions: Array of frame possessions
    @param reverse: if True, then look after `frame_idx` rather than before
    @param ignore_ids: if provided, ignore these ids
    @param window_size: return the mode of the last `window_size` possessions
    @return holder_id: ID for the holder (returns none if not found)
    @return last_possession_idx: Frame index for the last possession (returns none if not found)
    """
    step = 1 if reverse else -1
    start, end = (
        (frame_idx + 1, len(frame_possessions)) if reverse else (frame_idx - 1, -1)
    )
    for i in range(start, end, step):
        if (frame_possessions[i] is not None) and (
            frame_possessions[i] not in ignore_ids
        ):
            if window_size == 1:
                return frame_possessions[i], i
            # Guaranteed that window has at least one non-None element
            if reverse:
                window = frame_possessions[
                    i : min(i + window_size, len(frame_possessions))
                ]
            else:
                window = frame_possessions[max(i - window_size + 1, 0) : i + 1]
            try:
                # Find the mode that is not `None`
                holder_id = mode([x for x in window if x is not None])
            except StatisticsError:
                return frame_possessions[i], i
            if reverse:
                # Find first index after i where frame_possessions[i] = holder_id
                for j in range(len(window)):
                    if frame_possessions[i + j] == holder_id:
                        return frame_possessions[i + j], i + j
            else:
                # Find first index before i where frame_possessions[i] = holder_id
                for j in range(len(window)):
                    if frame_possessions[i - j] == holder_id:
                        return frame_possessions[i - j], i - j
    return None, None


def find_first_possession_from_index(
    frame_idx: int, frame_possessions: List[Optional[int]]
) -> int:
    r"""Find the first index that the current holder at `frame_idx` is holding the ball.
    :param frame_idx:
    :param frame_possessions: Array of frame possessions
    :return: Frame index for the first possession
    """
    holder_id = frame_possessions[frame_idx]
    assert holder_id is not None, "The passed frame index must have a holder."

    first_idx = frame_idx
    for i in range(frame_idx - 1, -1, -1):
        first_idx = i

        # Punt if we have reached another ID
        if frame_possessions[i] != holder_id:
            break
    assert frame_possessions[first_idx + 1] == holder_id, (
        f"Something is wrong - first_idx {first_idx} does not have the holder {holder_id}"
    )
    return first_idx + 1


def detect_shot_outcome(
    frame_names: List[str],
    keyframe_height: int,
    keyframe_width: int,
    last_possession_idx: int,
    ball_id: int,
    hoop_id: int,
    video_segments: SAM2SegmentsType,
    video_boxes: SAM2BoxesType,
) -> Tuple[bool, Optional[int], Optional[int]]:
    """We infer two important indices for the key ball:
    - The frame index when the ball first enters the backboard area
    - The frame index when the ball is deemed either a miss or a make
    @param ball_id: The id for the key ball (current one in consideration). It does
                    not matter if multiple balls are present, this function only tracks one
    @param hoop_id: The id for the tracked hoop. We use this instead of a bounding box.
    @param last_possession_idx: The index for the last frame where the scorer has possession of ball.
    @param video_segments: Map from frame index to inferred SAM2 masks
    @param video_boxes: Map from frame index to inferred boxes
    @return last_above_frame_idx: Index of the last frame the ball is above the hoop
    @return first_below_frame_idx: Index of the first frame the ball is below the hoop
    """
    num_frames = len(frame_names)

    # For each keyframe idxs, track if ball is above hoop or below net
    is_above_hoop = np.zeros(num_frames)
    is_below_backboard = np.zeros(num_frames)
    is_outside_backboard = np.zeros(num_frames)  # width only
    is_intersecting_hoop = np.zeros(num_frames)
    is_missing_ball = np.zeros(num_frames)

    for frame_idx in range(last_possession_idx, num_frames):
        if ball_id not in video_segments[frame_idx]:
            continue

        if hoop_id not in video_segments[frame_idx]:
            continue

        ball = video_segments[frame_idx][ball_id]  # shape: 1 x h x w
        ball_box = video_boxes[frame_idx][ball_id]

        if ball_box is None:  # ball is not in scene
            is_missing_ball[frame_idx] = 1
            continue

        # Fetch hoop bounding box from the tracked mask
        hoop_box = video_boxes[frame_idx][hoop_id]

        if hoop_box is None:  # hoop is not in scene
            continue

        # The shot box is the area of interest around the hoop
        shot_box = get_shot_box_from_hoop_box(hoop_box, image_size=256)
        shot_mask = bounding_box_to_binary_mask(
            shot_box[0],
            shot_box[1],
            shot_box[2],
            shot_box[3],
            keyframe_height,
            keyframe_width,
        )

        # Hallucinate the backboard from the hoop and build a mask from it
        backboard_box = hallucinate_backboard_from_hoop(
            hoop_box, keyframe_height, keyframe_width, relaxed=True
        )

        # Check if the ball is above the hoop and in the shot box
        if bool(compute_mask_intersection(shot_mask, ball) > 0) and is_ball_above_hoop(
            ball_box, hoop_box, decision_point="top"
        ):
            is_above_hoop[frame_idx] = 1
        # Check if the ball is below the net
        elif is_ball_below_backboard(ball_box, backboard_box):
            is_below_backboard[frame_idx] = 1
        # Check if the ball is outside the width of backboard
        elif is_ball_outside_backboard(ball_box, backboard_box):
            is_outside_backboard[frame_idx] = 1
        # NOTE: this is not an `elif` b/c it could be true along with the branches above
        if compute_bbox_intersection(hoop_box, ball_box) > 0:
            is_intersecting_hoop[frame_idx] = 1

    # Interpolate the positioning
    is_above_hoop = interpolate_consecutive_numbers(
        is_above_hoop.tolist(),
        thres=3,
        check_fn=lambda x: x == 0,
    )
    is_below_backboard = interpolate_consecutive_numbers(
        is_below_backboard.tolist(),
        thres=3,
        check_fn=lambda x: x == 0,
    )
    is_outside_backboard = interpolate_consecutive_numbers(
        is_outside_backboard.tolist(),
        thres=3,
        check_fn=lambda x: x == 0,
    )
    is_intersecting_hoop = interpolate_consecutive_numbers(
        is_intersecting_hoop.tolist(),
        thres=3,
        check_fn=lambda x: x == 0,
    )

    # Find the last time the ball is above the hoop post last_possession_idx
    last_above_frame_idx = None
    for frame_idx in range(last_possession_idx, num_frames):
        if (is_above_hoop[frame_idx] == 0) and (last_above_frame_idx is not None):
            break
        elif is_above_hoop[frame_idx] == 1:
            last_above_frame_idx = frame_idx

    if last_above_frame_idx is None:
        # If we cannot find a single frame where the ball is above the hoop
        # then this is likely a pass and not a shot
        return False, None, None

    # Find the first time the ball is below / outside the hoop after that
    first_below_frame_idx = None
    for frame_idx in range(last_above_frame_idx + 1, num_frames):
        if is_below_backboard[frame_idx] == 1 or is_outside_backboard[frame_idx] == 1:
            first_below_frame_idx = frame_idx
            break

    ball_is_stuck = False
    if first_below_frame_idx is None:
        last_intersect_frame_idx = None
        # Find the last frame where the ball is in the rim
        # NOTE on purpose no +1 here
        for frame_idx in range(last_above_frame_idx, num_frames):
            if is_intersecting_hoop[frame_idx] == 1:
                last_intersect_frame_idx = frame_idx
        if last_intersect_frame_idx is not None:
            no_remaining_ball_found = True
            # Check that we are missing ball after this is always
            for frame_idx in range(last_intersect_frame_idx + 1, num_frames):
                if is_missing_ball[frame_idx] == 0:  # 0 means not missing
                    no_remaining_ball_found = False
                    break
            if no_remaining_ball_found:
                # We use the last intersecting frame as the first frame below!
                first_below_frame_idx = last_intersect_frame_idx
                ball_is_stuck = True

    if first_below_frame_idx is None:
        # Protection mechanism in case we are tracking a super long shot that doesn't bounce down
        return False, None, None

    # NOTE we made this a <= check in case of ball_is_stuck
    assert last_above_frame_idx <= first_below_frame_idx, (
        f"Something is wrong - last_above_frame_idx {last_above_frame_idx} < first_below_frame_idx {first_below_frame_idx}"
    )
    # Find the bounding boxes for the ball
    ball_above_box = video_boxes[last_above_frame_idx][ball_id]
    ball_below_box = video_boxes[first_below_frame_idx][ball_id]
    assert ball_above_box is not None and ball_below_box is not None
    # Find the bounding boxes for the hoop
    hoop_above_box = video_boxes[last_above_frame_idx][hoop_id]
    hoop_below_box = video_boxes[first_below_frame_idx][hoop_id]
    assert hoop_above_box is not None and hoop_below_box is not None
    # Compute the shift from below to above!
    hoop_above_center = get_box_center(*hoop_above_box[:4])
    hoop_below_center = get_box_center(*hoop_below_box[:4])
    shift = (
        hoop_above_center[0] - hoop_below_center[0],
        hoop_above_center[1] - hoop_below_center[1],
    )

    def shift_box(box: BoxType, shift: Tuple[int | float, int | float]) -> BoxType:
        """Apply a shift to a bounding box.
        @note: hoop_below_center+(hoop_above_center-hoop_below_center) = hoop_above_center
        """
        return [
            box[0] + shift[0],
            box[1] + shift[1],
            box[2] + shift[0],
            box[3] + shift[1],
        ]

    if ball_is_stuck:
        # Heuristic - if we deem the ball is stuck in the rim, then mark it as in
        shot_outcome_by_ball = True
    else:
        # Heuristic - check if the path of the ball looks in
        # Take care to shift the below frame to the above frame coordinates
        shot_outcome_by_ball = did_ball_go_through_hoop(
            ball_above_box, shift_box(ball_below_box, shift), hoop_above_box
        )

        # If the ball is wider than the hoop above or below then this is a false positive
        if (ball_above_box[2] - ball_above_box[0]) > (
            hoop_above_box[2] - hoop_above_box[0]
        ):
            shot_outcome_by_ball = False
        if (ball_below_box[2] - ball_below_box[0]) > (
            hoop_above_box[2] - hoop_above_box[0]
        ):
            shot_outcome_by_ball = False

    return shot_outcome_by_ball, last_above_frame_idx, first_below_frame_idx


def create_shot_classification_video(
    video_path: str,
    hoops: List[HoopOutput],
    output_path: str,
    offset_ts: float = 0,
    image_size: int = 256,
    out_fps: int = 30,
):
    """Shot classification is a video classification model that takes a sequence of cropped frames
    as input. If we assume a static video, we can just use `truncate_and_crop_video` to do this with FFMPEG.
    However, since the camera may move, we cannot assume a static cropping.
    @param video_path: Path to the video file
    @param hoops: List of detected hoops
    @param start_ts: Start timestamp of the video segment
    @param end_ts: End timestamp of the video segment
    @param output_path: Path to save the output video
    @param image_size: Size of the cropped frames
    @param out_fps: Output video frame rate
    """
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video {video_path}")

    # Set the starting position in milliseconds
    fourcc = cv2.VideoWriter.fourcc(*"H264")  # Try H264 first
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (image_size, image_size))

    if not writer.isOpened():
        print("Could not open video writer in H264 format. Retrying with mp4v...")
        # Fallback to mp4v if H264 fails
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (image_size, image_size))

    if not writer.isOpened():
        raise Exception(f"Error: Could not open video writer for {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cur_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        hoop = estimate_hoop_at_timestamp(hoops, cur_ts + offset_ts)
        shot_box = get_shot_box_from_hoop_box(hoop.box, image_size=256)

        # Crop the frame using the shot box
        cropped_frame = frame[shot_box[1] : shot_box[3], shot_box[0] : shot_box[2]]
        writer.write(cropped_frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


@torch.no_grad()
def infer_person_keypoints(
    model,
    video_boxes: SAM2BoxesType,
    player_ids: List[int],
    frame_dir: str,
    frame_names: List[str],
    batch_size: int = 32,
    min_match_iou: float = 0.7,
) -> SAM2SkeletonsType:
    """Predict person keypoints
    @param player_ids: IDs for players
    @param frame_dir: Directory where frame images are saved
    @param frame_names: Filenames in the directory
    @param batch_size: Batch size to process images to (default: 32)
    @return video_skeletons:
    """
    video_skeletons: SAM2SkeletonsType = {}

    num_batches = len(frame_names) // batch_size + (len(frame_names) % batch_size != 0)
    for i in tqdm(range(num_batches), desc="predicting skeletons"):
        start_idx = batch_size * i
        end_idx = batch_size * (i + 1)

        batch_names = frame_names[start_idx:end_idx]
        batch_paths = [join(frame_dir, name) for name in batch_names]
        batch_bgr = np.stack([load_image(image_path) for image_path in batch_paths])
        batch_preds = model(batch_bgr)

        for j, frame_idx in enumerate(range(start_idx, end_idx)):
            if frame_idx not in video_boxes:
                continue

            # Fetch segments and bounding boxes frame-by-frame
            object_boxes = video_boxes[frame_idx]
            person_boxes = batch_preds[j]["instances"].pred_boxes.tensor.cpu().numpy()
            person_keypoints = batch_preds[j]["instances"].pred_keypoints.cpu().numpy()

            if len(person_boxes) == 0:
                video_skeletons[frame_idx] = {}
            else:
                # Need to disambiguate who is who
                # `id_map` maps a player ID in segments -> index in boxes/keypoints
                id_map = map_objects_to_players(
                    object_boxes, person_boxes, player_ids, min_iou=min_match_iou
                )
                # Do mapping from segment ID
                segment_keypoints: Dict[int, "npt.NDArray"] = {}
                for segment_id, box_idx in id_map.items():
                    segment_keypoints[segment_id] = person_keypoints[box_idx]
                video_skeletons[frame_idx] = segment_keypoints

    return video_skeletons


def map_objects_to_players(
    object_boxes: Dict[int, Optional[BoxType]],
    person_boxes: npt.NDArray,
    player_ids: List[int],
    min_iou: float = 0.7,
) -> Dict[int, int]:
    """Create map from segment ID to index in boxes.
    @param object_boxes: Map from player ID to bounding box
    @param person_boxes: List of predicted person boxes
    @param player_ids: List of player IDs
    @param min_iou: Minimum matching IOU to design a match (default: 0.7)
    """
    matching: Dict[int, int] = {}
    used_idxs: Set[int] = set()

    for player_id in player_ids:
        if player_id not in object_boxes:
            continue
        object_box = object_boxes[player_id]
        if object_box is None:
            continue
        object_box = np.array(object_box)

        ious = []
        for i in range(len(person_boxes)):
            if i in used_idxs:
                ious.append(0)
            else:
                iou = compute_bbox_iou(object_box.tolist(), person_boxes[i][:4])
                ious.append(iou)

        best_idx = int(np.argmax(ious))
        if ious[best_idx] < min_iou:
            continue

        used_idxs.add(best_idx)
        matching[player_id] = best_idx

    return matching


@torch.no_grad()
def embed_player_trajectory(
    model: SoliderFeatureExtractor,
    frame_dir: str,
    frame_names: List[str],
    segments: SAM2SegmentsType,
    boxes: SAM2BoxesType,
    keypoints: SAM2SkeletonsType,
    start_frame_idx: int,
    end_frame_idx: int,
    player_id: int,
    num_samples: int = 10,
    num_candidates: int = 50,
    min_keypoint_prob: float = 0.1,
    min_keypoint_present: int = 10,
    min_bbox_coverage: float = 0.2,
    top_k: int = 3,
    out_dir: Optional[str] = None,
) -> Optional[PlayerEmbedding]:
    """Sample a list of frame indices from the full segments for individual player.
    This is important to sample tag images and remove background.
    @param frame_dir: Directory of frame images
    @param frame_names: List of names
    @param segments: SAM2 segments
    @param boxes: bounding boxes
    @param keypoints: Person segments
    @param start_frame_idx: Frame index to start sampling
    @param end_frame_idx: Frame index to stop sampling
    @param player_id: Object id
    @param num_samples: Number of samples (default = 10)
    @param num_candidates: Number of candidates (default = 50)
    @param min_keypoint_prob: Minimum confidence to count a keypoint (default = 0.1)
    @param min_keypoint_present: Minimum number of keypoints present (default = 10)
    @param top_k: Number of embeddings to use for aggregation (default = 3)
    @param out_dir: If not None, then save the images created to this folder
    @return output:
        @key images: list of cropped images
        @key embs: list of embeddings in the same order
        @key feat: summarized vector for all images
    """
    if out_dir is not None:
        makedirs(out_dir, exist_ok=True)

    # Sample broadly which frames to look up
    frame_idxs = np.linspace(
        start_frame_idx,
        end_frame_idx,
        num=min(end_frame_idx - start_frame_idx - 1, num_candidates),
    ).tolist()
    frame_idxs = [int(x) for x in frame_idxs]

    # Score each frame by mask size
    player_idxs: List[int] = []
    player_num_keypts: List[int] = []
    player_coverages: List[float] = []
    for frame_idx in frame_idxs:
        # Ignore if player is not found in segments
        if player_id not in segments[frame_idx]:
            continue
        # Ignore if player is not found in keypoints
        if player_id not in keypoints[frame_idx]:
            continue
        # Compute of number of keypoints present
        player_keypoints = keypoints[frame_idx][player_id]
        num_keypoints_present = int(np.sum(player_keypoints[:, 2] >= min_keypoint_prob))
        # Punt if missing too many keypoints
        if num_keypoints_present < min_keypoint_present:
            continue
        # Compute the number of pixels showing in mask
        player_mask = segments[frame_idx][player_id][0]
        player_bbox = boxes[frame_idx][player_id]
        if player_bbox is None:
            continue
        # Get number of pixels present in mask
        num_pixels_present = int(np.sum(player_mask > 0))
        # Get total box size
        box_width = player_bbox[2] - player_bbox[0]
        box_height = player_bbox[3] - player_bbox[1]
        box_area = box_width * box_height
        # Punt if missing too many pixels
        if num_pixels_present / box_area < min_bbox_coverage:
            continue
        player_idxs.append(int(frame_idx))
        player_num_keypts.append(num_keypoints_present)
        player_coverages.append(num_pixels_present / box_area)

    if len(player_idxs) > num_samples:
        sample_idxs = np.linspace(0, len(player_idxs) - 1, num_samples).astype(int)
        player_idxs = [int(player_idxs[int(i)]) for i in sample_idxs]
        player_num_keypts = [int(player_num_keypts[int(i)]) for i in sample_idxs]

    if len(player_idxs) == 0:
        return None

    cropped_ims: List[np.ndarray] = []
    cropped_names: List[str] = []
    cropped_num_keypts: List[int] = []
    for i, frame_idx in tqdm(
        enumerate(player_idxs), desc=f"sampling for player {player_id}"
    ):
        if player_id not in segments[frame_idx]:
            continue

        num_keypts = player_num_keypts[i]

        # Get the mask and bounding box
        player_mask = segments[frame_idx][player_id][0]
        player_bbox = boxes[frame_idx][player_id]

        if player_bbox is None:
            continue

        # Load the image
        frame_path = join(frame_dir, frame_names[frame_idx])
        im = Image.open(frame_path)

        # Apply the mask
        im = np.asarray(im)

        # Remove the background
        masked_im = im * player_mask.astype(np.uint8)[..., np.newaxis]
        masked_im = Image.fromarray(masked_im)

        # Crop to bounding box
        cropped_im = masked_im.crop(
            (
                player_bbox[0],
                player_bbox[1],
                player_bbox[2],
                player_bbox[3],
            )
        )

        if out_dir is not None:
            cropped_im.save(join(out_dir, f"{frame_idx}.jpg"))

        cropped_im = np.asarray(cropped_im)
        cropped_ims.append(cropped_im)
        cropped_names.append(f"{frame_idx}.jpg")
        cropped_num_keypts.append(num_keypts)

    if len(cropped_ims) == 0:
        return None

    with torch.inference_mode(), torch.autocast("cuda"):
        # Compute embeddings for each image
        embs = model(cropped_ims).cpu()

    # Combine these together
    emb = k_max_pool(embs.unsqueeze(0), k=top_k).squeeze(0)

    # Choose the feature image as the one with most keypoints
    feat_name = cropped_names[int(np.argmax(cropped_num_keypts))]

    output = {
        "person_id": player_id,
        "frame_idxs": player_idxs,
        "image_names": cropped_names,
        "feat_name": feat_name,
        "embs": embs.numpy().tolist(),
        "feat": emb.numpy().tolist(),
    }
    output = PlayerEmbedding(**output)

    return output


def infer_clusters(
    all_shot_idxs: List[int],
    all_embeddings: List[List[PlayerEmbedding]],
    logger: logging.Logger,
    min_samples: int = 10,
) -> Tuple[List[ClusterOutput], List[int]]:
    """Infer clusters of persons across clips in a video.
    @param all_shot_idxs: List of length |# clips|.
                          Each contains the idx for that clip.
    @param all_embeddings: List of length |# clips|.
                           Each entry contains a list of embeddings for a sequence of images for each identified player in a clip.
    @param min_cluster_samples (default: 10): Minimum number of samples to define a neighborhood
    @return: List of cluster info
        shot_idx   - Index of the shot in this cluster
        person_id  - Identifier for the person (default: None)
        action_id  - Identifier for the action
        cluster_id - Identifier for the cluster
        image_name - Name of the local image file (default: None)
        image_url  - URL for the image URL (default: None)
    @return clusters:
    """
    embeddings = []  # size: |# shots|x|# persons per shot|x|# images per person|, embeddings for images sequences for each person per clip
    shot_idxs = []  # size: |# shots|x|# persons per shot|, idx for each shot/clip
    action_ids = []  # size: |# shots|x|# persons per shot|, id for each action
    tag_artifacts = []  # size: |# shots|x|# persons per shot|
    unique_ids = []  # size: |# shots|x|# persons per shot|, unique ids for each person per shot
    count = 0  # counter for unique ids
    skipped_idxs = []  # list of indices of skipped shots (those without identified scorer)
    for i in range(len(all_embeddings)):
        shot_idx = all_shot_idxs[i]
        person_embeddings = all_embeddings[i]
        if len(person_embeddings) == 0:
            logger.info(f"embeddings not found for shot {i}. skipping...")
            skipped_idxs.append(shot_idx)
            continue
        for e in person_embeddings:
            if e.tag_artifact is None:
                continue
            assert e.person_id == e.tag_artifact.id
            snapshot_embeddings = np.asarray(e.embs)
            embeddings.append(snapshot_embeddings)
            shot_idxs.append(shot_idx)
            action_ids.append(e.action_id)
            tag_artifacts.append(e.tag_artifact)
            unique_ids.append(np.ones(len(snapshot_embeddings)) * count)
            count += 1

    clusters: List[ClusterOutput] = []
    categories = []
    if len(embeddings) > 0:
        embeddings = np.vstack(embeddings)
        unique_ids = np.concatenate(unique_ids)

        # Cluster embeddings across the different shots
        # assignments - size: |# shots|x|# persons per shot|x|# images per person|
        assignments, categories = cluster_embedding_neighborhoods(
            embeddings, min_samples=min_samples
        )
        # Find the most common assignment per person
        # assignments - size: |# shots|x|# persons per shot|
        assignments = find_most_common_assignment(assignments, unique_ids)

        assert len(assignments) == len(shot_idxs) == len(action_ids), (
            f"Expected |assignments|=|shot_idxs|=|action_ids|. Got {len(assignments)},{len(shot_idxs)},{len(action_ids)}"
        )
        # Build cluster objects
        for i in range(len(assignments)):
            tag: TagArtifact = tag_artifacts[i]
            cluster = {
                "shot_idx": int(shot_idxs[i]),
                "person_id": int(tag.id),
                "action_id": int(action_ids[i]),
                "cluster_id": int(assignments[i]),
                "image_name": tag.name,
                "image_url": tag.url,
            }
            cluster = ClusterOutput(**cluster)
            clusters.append(cluster)

    # Add the -1 cluster if we have it for some reason
    if -1 not in categories:
        categories.append(-1)

    # Add any missing shot idxs as -1 cluster
    for shot_idx in skipped_idxs:
        cluster = {
            "shot_idx": int(shot_idx),
            "person_id": None,
            "action_id": ACTION_SHOT,
            "cluster_id": -1,
            "image_name": None,
            "image_url": None,
        }
        cluster = ClusterOutput(**cluster)
        clusters.append(cluster)

    return clusters, categories


def cluster_embedding_neighborhoods(
    embeddings: npt.NDArray, min_samples: int = 10
) -> Tuple[npt.NDArray, List[int]]:
    """Cluster embeddings into buckets.
    @param embeddings: shape nxd - all embeddings for users across many tag images
    @param min_samples: Minimum number of samples to define a neighborhood (default: 10)
    @return assignments:
    @note: The bucket -1 will be reserved for images that do not fit in
    """
    # DBScan does not work well in high dimensions where distance makes less sense
    if len(embeddings) > 50:
        # Project to two dimensions using TSNE - much better than PCA
        tsne = TSNE(n_components=2)
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        # If too few data points, we need to use PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    # We also want to normalize the distances
    # This is important for DBScan
    scaler = StandardScaler()
    embeddings_2d = scaler.fit_transform(embeddings_2d)
    # Estimate a good epsilon by grid search
    eps_hat = estimate_eps_silhouette(embeddings_2d, min_samples=min_samples)
    # Fit the model
    model = DBSCAN(eps=eps_hat, min_samples=min_samples)
    model.fit(embeddings_2d)
    # Get assignments
    assignments = np.asarray(model.labels_)
    categories = np.unique(assignments).tolist()

    return assignments, categories


def find_most_common_assignment(
    assignments: npt.NDArray, object_ids: npt.NDArray
) -> npt.NDArray:
    """Find the most common assignment per person."""
    outputs = []
    unique_ids = np.unique(object_ids)
    for unique_id in unique_ids:
        most_common = stats.mode(assignments[object_ids == unique_id]).mode
        outputs.append(most_common)

    return np.asarray(outputs)


def estimate_eps_silhouette(embeddings: npt.NDArray, min_samples: int = 10) -> float:
    """Estimate the best eps parameter for DBSCAN using the silhouette score.
    @param embeddings: shape nxd - all embeddings for users across many tag images
    @param min_samples: Minimum number of samples to define a neighborhood (default: 10)
    @return: Best eps parameter
    """
    eps_values = np.arange(0.1, 1.0, 0.05)
    best_eps = None
    best_silhouette = -1
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(embeddings)
        # Filter out noise points (-1) for silhouette score calculation
        if (
            len(set(labels)) > 1
        ):  # Silhouette score is undefined if only 1 cluster or all noise
            silhouette = silhouette_score(embeddings, labels)
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_eps = eps
    if best_eps is None:
        best_eps = 0.5
    return best_eps


def get_shot_box_from_hoop_box(hoop_box: BoxType, image_size: int) -> BoxType:
    """Infer shot box from hoop box.
    Used in inference pipeline and training pipelines.
    """
    x1, y1, x2, y2 = hoop_box
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    # Take care to ensure that the output is exactly (image_size, image_size)
    # since this is used for video creation
    x1_ = max(int(xc - image_size // 2), 0)
    y1_ = max(int(yc - image_size // 2), 0)
    x2_ = x1_ + image_size
    y2_ = y1_ + image_size
    box = [x1_, y1_, x2_, y2_]
    return box
