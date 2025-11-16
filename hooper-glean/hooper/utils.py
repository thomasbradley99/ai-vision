import re
import cv2
import uuid
import yaml
import json
import math
import time
import ffmpeg
import torch
import joblib
import shutil
import logging
import jsonlines
import numpy as np
import scipy as sp
from tqdm import tqdm
from datetime import datetime
import numpy.typing as npt
from matplotlib.axes import Axes
from collections import defaultdict
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageType
import matplotlib.patches as patches
import supervision as sv
import matplotlib.pyplot as plt
from os import makedirs, listdir, remove
from os.path import (
    splitext, exists, join, realpath, dirname, isdir, isfile
)
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial.distance import cdist

from shared_utils.types import (
    BoxType,
    PointType,
    PromptType,
    CheckFunctionType,
    SAM2SegmentsType,
    SAM2BoxesType,
    CompressedSegmentsType,
    HoopOutput,
)

# Action IDs
ACTION_SHOT: int = 1
ACTION_REBOUND: int = 2
ACTION_ASSIST: int = 3


def get_root_dir() -> str:
    return realpath(join(dirname(__file__), ".."))


def get_checkpoints_dir() -> str:
    root_dir = get_root_dir()
    return realpath(join(root_dir, "checkpoints"))


def get_firebase_service_account_path() -> str:
    root_dir = get_root_dir()
    return realpath(join(root_dir, "hooper-service-account.json"))


def get_config_path() -> str:
    root_dir = get_root_dir()
    return realpath(join(root_dir, "config.yaml"))



CONFIG_FILE = get_config_path()
COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def get_coco_keypoint_index(name: str) -> int:
    assert name in COCO_KEYPOINTS, f"Unsupported coco keypoints: {name}"
    return COCO_KEYPOINTS.index(name)


def rgb_to_bgr(x: npt.NDArray) -> npt.NDArray:
    """Convert RGB image to BGR image."""
    return x[..., ::-1].copy()


def get_optical_flow_magnitude(im0: ImageType, im1: ImageType) -> float:
    """Get a measure of movement using optical flow.
    @param im0 (PIL.Image): first image
    @param im1 (PIL.Image): second image
    @return magnitude (float): Median magnitude
    """
    im0_ = np.asarray(im0)
    im1_ = np.asarray(im1)
    flow = cv2.calcOpticalFlowFarneback(im0_, im1_, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # type: ignore
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.median(magnitude))


def build_positive_clicks_from_skeleton(
    keypoints: npt.NDArray,
    min_score: float = 0.3,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Discern a few positive clicks for a predicted skeleton
    @param keypoints: torch.Tensor (17x3)
    @return points: np.ndarray (nx2)
    @return labels: np.ndarray (n)
    """
    points = []
    labels = []
    for i in range(len(keypoints)):
        x, y, score = keypoints[i]
        if score >= min_score:
            points.append([x, y])
            labels.append(1)
    points = np.array(points, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return points, labels


def build_positive_clicks_from_ball(ball: BoxType) -> Tuple[npt.NDArray, npt.NDArray]:
    x1, y1, x2, y2, _ = ball
    points = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    return points, labels


def find_index_of_closest_timestamp(timestamps: List[float], search_ts: float) -> int:
    """Find index of closest timestamp to a search timestamp.
    @param timestamps: List of timestamps (floats) sorted in ascending order.
    @param search_ts: The timestamp (float) for which we are searching the closest match.
    @return idx: The index of the timestamp in the list that is closest to search_ts.
    """
    # Initialize the index and the minimum difference
    closest_idx = 0
    min_diff = float("inf")
    # Iterate through the list to find the closest timestamp
    for i, ts in enumerate(timestamps):
        diff = abs(ts - search_ts)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i
    return closest_idx


def sample_frame_at_timestamp(
    video_file: str, key_timestamp: float, out_file: str, quiet: bool = False
):
    """Extract a keyframe at a timestamp using FFMPEG."""
    ffmpeg.input(video_file, ss=round(key_timestamp, 2)).output(
        out_file, vframes=1
    ).run(overwrite_output=True, quiet=quiet)


def sample_frames_from_video(
    video_file: str, out_dir: str, stride: int = 1, start: int = 0
) -> bool:
    """Sample exhaustively from video to an output directory."""
    makedirs(out_dir, exist_ok=True)
    frames_generator = sv.get_video_frames_generator(
        video_file, stride=stride, start=start
    )
    sink = sv.ImageSink(target_dir_path=out_dir, image_name_pattern="{:05d}.jpg")
    with sink:
        for frame in frames_generator:
            sink.save_image(frame)
    return True


def ffmpeg_get_creation_time(video_file: str) -> Optional[float]:
    """Extract the creation timestamp from a video file using ffmpeg."""
    try:
        # Probe the video file for metadata
        probe = ffmpeg.probe(video_file)

        # Look for creation time in format metadata
        format_metadata = probe.get("format", {})
        format_tags = format_metadata.get("tags", {})

        # Check common creation time fields
        creation_time = (
            format_tags.get("creation_time")
            or format_tags.get("date")
            or format_tags.get("DATE")
            or format_tags.get("Creation Time")
        )

        # If not found in format, check streams
        if not creation_time:
            for stream in probe.get("streams", []):
                stream_tags = stream.get("tags", {})
                creation_time = (
                    stream_tags.get("creation_time")
                    or stream_tags.get("date")
                    or stream_tags.get("DATE")
                )
                if creation_time:
                    break

        # Parse the timestamp if found
        if creation_time:
            # Handle common timestamp formats
            if creation_time.endswith("Z"):
                # ISO format with Z timezone
                result = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
                return int(result.timestamp())
            else:
                # Try parsing as ISO format
                result = datetime.fromisoformat(creation_time)
                return int(result.timestamp())

        return None

    except Exception as e:
        print(f"Error extracting creation time: {e}")
        return None


def move_subsampled_frames_out(in_dir: str, out_dir: str, stride: int = 1) -> bool:
    """Move & and rename some of the frames from `in_dir` to `subsampled` by stride.
    @param stride: Sampling rate between frames (default: stride)
    """
    makedirs(out_dir, exist_ok=True)
    frame_names = get_sampled_frame_names(in_dir)  # get sampled frames
    for frame_name in frame_names[::stride]:  # pick every other image
        shutil.move(join(in_dir, frame_name), join(out_dir, frame_name))
    return True


def merge_subsampled_frames_back(in_dir: str, out_dir: str) -> bool:
    """Move files back into `out_dir`."""
    assert isdir(in_dir) and isdir(out_dir), "directories do not exist"
    frame_names = get_sampled_frame_names(in_dir)
    for frame_name in frame_names:
        shutil.move(join(in_dir, frame_name), join(out_dir, frame_name))
    return True


def get_sampled_frame_names(frame_dir: str) -> List[str]:
    """Reads the file names assuming you have used `sample_frames_from_video`.
    @param frame_dir: Directory to read from
    @return: Sampled frames
    """
    frame_names = [
        p
        for p in listdir(frame_dir)
        if splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(splitext(p)[0]))
    return frame_names


def load_image(image: str, as_rgb: bool = False) -> npt.NDArray:
    """Read the image into a NumPy array.
    @param image: Path to an image
    @param as_rgb: Return RGB or a BGR image.
    """
    im_pil = Image.open(image).convert("RGB")
    im_rgb = np.asarray(im_pil)
    if as_rgb:
        return im_rgb
    im_bgr = rgb_to_bgr(im_rgb)
    return im_bgr


def load_grayscale_image(image: str) -> npt.NDArray:
    im_pil = Image.open(image).convert("L")
    im_gray = np.asarray(im_pil)
    return im_gray


def show_mask(
    mask: npt.NDArray,
    ax: Axes,
    obj_id: Optional[int] = None,
    random_color: bool = False,
):
    """Plot the mask on the figure."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(
    coords: npt.NDArray, labels: npt.NDArray, ax: Axes, marker_size: int = 200
):
    """Plot some markers on the figure."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_bbox(
    box: BoxType,
    ax: Axes,
    color: str,
    label: str,
    linewidth: int = 2,
    linestyle: str = "solid",
):
    """Plot a bounding box on the figure."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    # Add rectangle to plot
    rect = patches.Rectangle(
        (x1, y1),
        w,
        h,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    # Add label text
    ax.text(x1, y1, label, fontsize=8, color=color, verticalalignment="top")


def show_skeleton(keypoints: npt.NDArray, ax: Axes):
    """Plot person skeleton on image."""
    # skeleton COCO format
    skeleton = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [5, 7],
        [7, 9],
        [6, 8],
        [8, 10],
        [5, 6],
        [5, 11],
        [6, 12],
        [11, 12],
        [11, 13],
        [12, 14],
        [13, 15],
        [14, 16],
    ]
    # Plot person keypoints
    for i in range(len(keypoints)):
        x, y, v = keypoints[i]
        if v > 0:
            ax.plot(x, y, "o", color="blue", markersize=5)
    # Plot skeleton connections
    for edge in skeleton:
        x0, y0, v0 = keypoints[edge[0]]
        x1, y1, v1 = keypoints[edge[1]]
        if v0 > 0 and v1 > 0:
            ax.plot([x0, x1], [y0, y1], color="red", linewidth=2)


def compute_iou(mask1: npt.NDArray, mask2: npt.NDArray) -> float:
    r"""Computes the Intersection over Union (IoU) between two binary masks.
    :param mask1: First binary mask
    :param mask2: Second binary mask
    :return iou: IoU score
    """
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    # Compute IoU
    if union == 0:
        return 0.0  # Avoid division by zero
    iou = intersection / union
    return iou


def compute_mask_intersection(mask1: npt.NDArray, mask2: npt.NDArray) -> float:
    """Cheaper alternative to compute_iou if we only care about intersection."""
    intersection = np.logical_and(mask1, mask2).sum()
    return intersection


def compute_bbox_intersection(bbox1: BoxType, bbox2: BoxType) -> float:
    """Returns the intersection area between two bounding boxes."""
    x11, y11, x21, y21 = bbox1[:4]
    x12, y12, x22, y22 = bbox2[:4]
    intersection = max(0, min(x21, x22) - max(x11, x12)) * max(
        0, min(y21, y22) - max(y11, y12)
    )
    return intersection


def compute_bbox_iou(bbox1: BoxType, bbox2: BoxType) -> float:
    r"""Implement the intersection over union (IoU) between box1 and box2"""
    # Compute edges of bboxes
    x11, y11, x21, y21 = bbox1[:4]
    x12, y12, x22, y22 = bbox2[:4]
    yi1 = max(y11, y12)
    xi1 = max(x11, x12)
    yi2 = min(y21, y22)
    xi2 = min(x21, x22)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)
    union_area = box1_area + box2_area - inter_area
    # compute the IoU
    iou = inter_area / union_area
    return iou


def compute_bbox_area(bbox: BoxType) -> float:
    """Implement the bbox area."""
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return area


def filter_consecutive_numbers(lst: List, thres: int = 3) -> List:
    """Set indices to None unless there are a minimum of `thres` consecutive appearances.
    @param used in possession classification
    """
    result = lst[:]
    n = len(lst)
    i = 0
    while i < n:
        if lst[i] is not None:
            count = 1
            while i + count < n and lst[i + count] == lst[i]:
                count += 1
            if count < thres:
                for j in range(count):
                    result[i + j] = None
            i += count
        else:
            i += 1
    return result


def interpolate_consecutive_numbers(
    lst: List,
    thres: int = 3,
    check_fn: CheckFunctionType = lambda x: x is None,
) -> List:
    """Interpolate a sequence of integers to replace None entries where
    the digit before and after the Nones are identical and there are no more than
    `thres` None elements in a row.
    @param lst:
    @param thres: Maximum number of allowable consecutive None to interpolate between
    """
    result = lst[:]
    n = len(lst)
    i = 0
    while i < n:
        if check_fn(result[i]):
            start = i
            while i < n and check_fn(result[i]):
                i += 1
            end = i
            if (
                end - start < thres
                and start > 0
                and end < n
                and result[start - 1] == result[end]
            ):
                for j in range(start, end):
                    result[j] = result[start - 1]
        else:
            i += 1
    return result


def get_box_center(
    x1: int | float,
    y1: int | float,
    x2: int | float,
    y2: int | float,
) -> Tuple[int | float, int | float]:
    """Get the center coordinate for box."""
    xc = int((x1 + x2) / 2.0)
    yc = int((y1 + y2) / 2.0)
    return xc, yc


def bounding_box_to_binary_mask(
    x1: int | float,
    y1: int | float,
    x2: int | float,
    y2: int | float,
    h: int | float,
    w: int | float,
) -> npt.NDArray:
    """Convert a bounding box to a binary mask.
    :param x1, y1, x2, y2 (int): Coordinates of the bounding box (x1 <= x2, y1 <= y2)
    :param h (int): Height of the image
    :param w (int): Width of the image
    :return numpy.ndarray: Binary mask of shape (h, w) with 1s inside the bounding box and 0s elsewhere
    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Initialize a binary mask with zeros
    mask = np.zeros((h, w), dtype=np.uint8)  # type: ignore

    # Ensure bounding box coordinates are within the image dimensions
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # Set the region inside the bounding box to 1
    mask[y1:y2, x1:x2] = 1

    return mask.astype(bool)


def binary_mask_to_bounding_box(mask: npt.NDArray) -> Optional[BoxType]:
    """Convert a binary mask to a bounding box
    @param np.ndarray:
    @return x1, y1, x2, y2:
    """
    mask = mask.astype(int)  # binary - 0/1 mask
    # Check if mask is empty (all zeros)
    if not np.any(mask):
        return None
    # Find non-zero elements directly
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    # Find the minimum and maximum coordinates
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    # Return the bounding box coordinates, +1 to make it inclusive
    bbox = [x_min, y_min, x_max + 1, y_max + 1]
    return format_bbox(bbox)


def is_ball_above_hoop(
    ball: BoxType, hoop: BoxType, decision_point: str = "midpoint"
) -> bool:
    """Check if a ball is above the hoop.
    @param ball: Ball bounding box
    @param hoop: Hoop bounding box
    @return (bool)
    """
    ball_y1 = ball[1]
    ball_y2 = ball[3]
    hoop_y1 = hoop[1]
    if decision_point == "midpoint":
        # Check if the midpoint of the ball is above the top of the hoop
        return (ball_y1 + ball_y2) / 2.0 < hoop_y1
    elif decision_point == "top":
        # Check if top of ball is above the top of the hoop
        return ball_y1 < hoop_y1
    else:
        # Check if bottom of ball is above the top of the hoop
        return ball_y2 < hoop_y1


def is_ball_below_backboard(ball: BoxType, backboard: BoxType) -> bool:
    """Check if the top of the ball is below the bottom of the backboard.
    @param ball: Ball bounding box
    @param backboard: Backboard bounding box
    @return (bool)
    """
    ball_y1 = ball[1]
    backboard_y2 = backboard[3]
    # Check if top of ball is below backboard bottom
    return ball_y1 > backboard_y2


def is_ball_outside_backboard(ball: BoxType, backboard: BoxType) -> bool:
    """If the ball floats to the left and right of the backboard.
    :param ball: Ball bounding box
    :param backboard: Backboard bounding box
    """
    ball_x1, _, ball_x2, _ = ball
    backboard_x1, _, backboard_x2, _ = backboard
    return (ball_x2 < backboard_x1) or (ball_x1 > backboard_x2)


def did_ball_go_through_hoop(
    ball_above: BoxType,
    ball_below: BoxType,
    hoop: BoxType,
) -> bool:
    """Check if a ball is in/through the hoop.
    :param ball_above: [x1, y1, x2, y2]
    :param ball_below: [x1, y1, x2, y2]
    :param hoop: [x1, y1, x2, y2]
    :return (bool): True if the ball is near the hoop, False otherwise
    """
    old_x1, old_y1, old_x2, old_y2 = ball_above[:4]
    ball_x1, ball_y1, ball_x2, ball_y2 = ball_below[:4]
    hoop_x1, hoop_y1, hoop_x2, hoop_y2 = hoop[:4]

    # If the old ball is below the current position, this cannot be a shot
    if (old_y1 > ball_y1) or (old_y2 > ball_y2):
        return False

    # If top of ball is above the bottom of hoop, then this is not a shot yet
    if ball_y1 <= hoop_y2:
        return False

    # If the ball is more than three ball heights below hoop, do not classify as shot
    ball_height = abs(ball_y2 - ball_y1)
    if ball_y1 > (hoop_y1 + ball_height * 3):
        return False

    # Build a line between old ball and new ball
    m, b = np.polyfit(
        [(old_x1 + old_x2) / 2.0, (ball_x1 + ball_x2) / 2.0],
        [(old_y1 + old_y2) / 2.0, (ball_y1 + ball_y2) / 2.0],
        1,
    )

    hoop_y = (hoop_y1 + hoop_y2) / 2  # middle y-coord of hoop
    # y = mx + b => x = (y - b) / m
    pred_x = (hoop_y - b) / m

    # Compute the width
    hoop_width = abs(hoop_x2 - hoop_x1)

    # Save 15% of the hoop as not valid
    hoop_buffer = hoop_width * 0.075

    if (hoop_x1 + hoop_buffer) < pred_x < (hoop_x2 - hoop_buffer):
        # Check if the predicted x coordinate in between the rim
        return True

    return False


def hallucinate_backboard_from_hoop(
    hoop_box: BoxType,
    image_height: int,
    image_width: int,
    relaxed: bool = False,
) -> BoxType:
    """Prediction for where the shot region / backboard is from the hoop.
    :param hoop_box: xyxy bounding box
    :param image_height:
    :param image_width:
    :param backboard_box: xyxy bounding box
    :param relaxed (default: False): whether to relax or not
    """
    hoop_x1, hoop_y1, hoop_x2, hoop_y2 = hoop_box
    hoop_width = hoop_x2 - hoop_x1
    hoop_height = hoop_y2 - hoop_y1
    if not relaxed:
        backboard_box = [
            hoop_x1 - hoop_width * 1.5,
            hoop_y1 - hoop_width * 2.0,
            hoop_x2 + hoop_width * 1.5,
            hoop_y2 + hoop_height * 1.5,
        ]
    else:
        backboard_box = [
            hoop_x1 - hoop_width * 3.0,
            0,
            hoop_x2 + hoop_width * 3.0,
            hoop_y2 + hoop_height * 1.5,
        ]
    # clamp the value
    backboard_box[0] = clamp_value(backboard_box[0], 0, image_width)
    backboard_box[2] = clamp_value(backboard_box[2], 0, image_width)
    backboard_box[1] = clamp_value(backboard_box[1], 0, image_height)
    backboard_box[3] = clamp_value(backboard_box[3], 0, image_height)
    # round the value
    backboard_box = [round(x, 3) for x in backboard_box]
    backboard_box = format_bbox(backboard_box)
    return backboard_box


def unflatten_boxes(boxes_flat: Union[List[float], List[int]]) -> List[BoxType]:
    """Format box annotations to a list of boxes.
    @param boxes_flat (List[float]) - flattened list of [x1, y1, x2, y2, ts, x1, y1, x2, y2, ts, ...]
    """
    assert len(boxes_flat) % 5 == 0, "Incorrect shape. Must be divisible by 5"
    num_boxes = len(boxes_flat) // 5
    boxes = []
    for i in range(num_boxes):
        box_i: BoxType = boxes_flat[i * 5 : (i + 1) * 5]
        boxes.append(box_i)
    return boxes


def flatten_boxes(boxes: List[BoxType]) -> Union[List[float], List[int]]:
    """Flatten boxes into a list."""
    output = []
    for i in range(len(boxes)):
        output.extend(boxes[i])
    return output


def unflatten_points(points_flat: Union[List[float], List[int]]) -> List[PointType]:
    assert len(points_flat) % 3 == 0, "Incorrect shape. Must be divisible by 3"
    num_points = len(points_flat) // 3
    points = []
    for i in range(num_points):
        box_i: PointType = points_flat[i * 3 : (i + 1) * 3]
        points.append(box_i)
    return points


def flatten_points(points: List[PointType]) -> Union[List[float], List[int]]:
    """Flatten points into a list."""
    output = []
    for i in range(len(points)):
        output.extend(points[i])
    return output


def unstringify_hoops(hoops_str: str) -> List[HoopOutput]:
    """Unstringify hoops into a list."""
    hoops = []
    for hoop_str in hoops_str.split(";"):
        x1, y1, x2, y2, ts = map(float, hoop_str.split(","))
        box = [int(x1), int(y1), int(x2), int(y2)]
        hoops.append(HoopOutput(ts=float(ts), box=box, prob=1.0))
    return hoops


def stringify_hoops(hoops: List[HoopOutput]) -> str:
    """Stringify hoops into a string."""
    hoops_str = []
    for hoop in hoops:
        hoops_str.append(
            f"{hoop.box[0]},{hoop.box[1]},{hoop.box[2]},{hoop.box[3]},{hoop.ts}"
        )
    return ";".join(hoops_str)


def find_hoop_neighbor_idxs_to_timestamp(
    hoops: List[HoopOutput], query_ts: float
) -> Tuple[int, int]:
    """Given a video timestamp, find the index of the hoop before and after the query timestamp.
    @param hoops: List of predicted hoops from inference pipeline.
    @param query_ts: Timestamp to use to find the right hoop.
    """
    if len(hoops) == 0:
        return (0, 0)
    # If query is before the first hoop
    if query_ts <= hoops[0].ts:
        return (0, 0)
    # If query is after the last hoop
    if query_ts >= hoops[-1].ts:
        last_idx = len(hoops) - 1
        return (last_idx, last_idx)
    # Binary search to find the insertion point
    left, right = 0, len(hoops) - 1
    while left <= right:
        mid = (left + right) // 2
        if hoops[mid].ts < query_ts:
            left = mid + 1
        else:
            right = mid - 1
    # At this point, right is the index of the last hoop before query_ts
    # and left is the index of the first hoop after query_ts
    before_idx = right
    after_idx = left
    return (before_idx, after_idx)


def interpolate_hoops(
    hoop1: HoopOutput, hoop2: HoopOutput, query_ts: float
) -> HoopOutput:
    """Interpolate between two hoops."""
    # Handle edge cases where timestamps are equal
    if hoop1.ts == hoop2.ts:
        return HoopOutput(ts=query_ts, box=hoop1.box, prob=hoop1.prob)

    # Calculate interpolation factor (0.0 to 1.0)
    t = (query_ts - hoop1.ts) / (hoop2.ts - hoop1.ts)
    t = max(0.0, min(1.0, t))  # Clamp to [0, 1]

    # Interpolate box coordinates if both hoops have boxes
    # Assuming BoxType is [x1, y1, x2, y2] or similar
    box1, box2 = hoop1.box, hoop2.box
    interpolated_box = [box1[i] * (1 - t) + box2[i] * t for i in range(len(box1))]

    # Interpolate probability if both hoops have probabilities
    interpolated_prob = None
    if hoop1.prob is not None and hoop2.prob is not None:
        interpolated_prob = hoop1.prob * (1 - t) + hoop2.prob * t
    elif hoop1.prob is not None:
        interpolated_prob = hoop1.prob
    elif hoop2.prob is not None:
        interpolated_prob = hoop2.prob

    return HoopOutput(ts=query_ts, box=interpolated_box, prob=interpolated_prob)


def find_closest_box_idx_to_timestamp(boxes: List[BoxType], query_ts: float) -> int:
    """Given a video timestamp, find the closest hoop detection.
    @param boxes: List of boxes to search through. We expect each of these boxes to be size [x1, y1, x2, y2, ts]
    @param query_ts: Timestamp to use to find the right hoop.
    @return idx: Index for the candidate
    """
    # Keep only the boxes with timestamps
    valid_boxes = [box for box in boxes if box is not None and len(box) == 5]
    if len(valid_boxes) == 0:
        return 0
    # Default to index 0
    dist_ts = abs(valid_boxes[0][4] - query_ts)
    closest_idx = 0
    # Search for the closest index
    for i in range(1, len(valid_boxes)):
        if abs(valid_boxes[i][4] - query_ts) < dist_ts:
            closest_idx = i
            dist_ts = abs(valid_boxes[i][4] - query_ts)
    return closest_idx


def find_closest_point_idx_to_timestamp(
    points: List[PointType], query_ts: float
) -> int:
    """Given a video timestamp, find the closest hoop detection.
    @param boxes: List of points to search through. We expect each of these points to be size [x, y, ts]
    @param query_ts: Timestamp to use to find the right point.
    @return idx: Index for the candidate
    """
    # Keep only the points with timestamps
    valid_points = [pt for pt in points if pt is not None and len(pt) == 3]
    if len(valid_points) == 0:
        return 0
    # Default to index 0
    dist_ts = abs(valid_points[0][2] - query_ts)
    closest_idx = 0
    # Search for the closest index
    for i in range(1, len(valid_points)):
        if abs(valid_points[i][2] - query_ts) < dist_ts:
            closest_idx = i
            dist_ts = abs(valid_points[i][2] - query_ts)
    return closest_idx


def find_closest_detection_idx_to_point(
    boxes: torch.Tensor, query_pt: PointType
) -> int:
    """Pick the box that is closest to the annotated point.
    @param point: [x,y]
    @return best_idx: Index for the best hoop. If there are no hoops, this returns 0
    """
    x0, y0 = query_pt
    # Find the default index
    best_idx = 0
    best_dist = math.inf
    # Compute the centers for each box
    for i in range(len(boxes)):
        box_i = boxes[i].cpu().numpy().tolist()
        if len(box_i) == 5:
            x1, y1, x2, y2, _ = box_i
        else:
            x1, y1, x2, y2 = box_i
        xc, yc = get_box_center(x1, y1, x2, y2)
        dist = math.sqrt((x0 - xc) ** 2 + (y0 - yc) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def clamp_value(x: float, min_x: float, max_x: float) -> float:
    r"""Utility function clamp a value."""
    return min(max(x, min_x), max_x)


def binary_mask_to_coco_format(mask: npt.NDArray) -> List[List[float]]:
    r"""Convert a binary mask to COCO segmentation format.
    :param mask: A 2D numpy array where the object is represented by 1s and the background by 0s.
    :return segmentation: list of contours
    """
    # Ensure binary mask is binary (0 or 1)
    mask = mask.astype(np.uint8)
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentations = []
    for contour in contours:
        # Flatten the contour array and convert it to a list of points
        segmentation = contour.flatten().tolist()
        segmentation = [int(round(x)) for x in segmentation]  # quantize
        segmentations.append(segmentation)
    return segmentations


def coco_format_to_binary_mask(
    paths: List[List[int]], image_width: int, image_height: int
) -> npt.NDArray:
    r"""Convert a COCO segmentation mask back to a binary mask.
    :return mask: h x w shape
    """
    # Create an empty binary mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    # Iterate over each segmentation contour
    for path in paths:
        # Convert the list of points back to the contour format (Nx1x2) required by OpenCV
        contour = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
        # Draw the contour on the mask with the value 1
        cv2.drawContours(mask, [contour], -1, color=1, thickness=cv2.FILLED)  # type: ignore
    return mask


def load_sklearn_model(model_path: str):
    r"""Load a trained sklearn model saved with joblib.
    :param model_path:
    """
    with open(model_path, "rb") as fp:
        clf = joblib.load(fp)
    return clf


def k_max_pool(features: torch.Tensor, k: int) -> torch.Tensor:
    """K-max pooling.
    https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_the_Best_Pooling_Strategy_for_Visual_Semantic_Embedding_CVPR_2021_paper.pdf

    8 ... 9 ... 4      9 ... 8 ... 4
    5 ... 3 ... 2  ->  5 ... 3 ... 2  ->  weighted sum
    5 ... 4 ... 7      7 ... 5 ... 4

    How do we learn the weights? Probably through labeled examples.
    Or we can just max K?

    :param features: features with shape B x K x D
    :return: pooled feature with shape B x D
    """
    # Sort along the given dimension in descending order
    sorted_vals, _ = torch.sort(features, dim=1, descending=True)
    pooled = sorted_vals[:, :k, :].sum(1) / float(k)
    return pooled


def compute_dist_ball_to_hands_fast(
    ball_box: BoxType,
    person_skeleton: npt.NDArray,
    min_conf: float = 0.7,
) -> Tuple[float, float]:
    """Similar to `compute_dist_ball_to_hands`. Since array checks are expensive
    we reduce the number of operations with a box check.
    """
    lh_y, lh_x, lh_pr = person_skeleton[get_coco_keypoint_index("left_wrist")]
    rh_y, rh_x, rh_pr = person_skeleton[get_coco_keypoint_index("right_wrist")]
    bb_x, bb_y = get_box_center(*ball_box)

    if lh_pr >= min_conf:
        lh_dist = math.sqrt((bb_x - lh_x) ** 2 + (bb_y - lh_y) ** 2)
    else:
        # If not confident in keypoint, then ignore
        lh_dist = math.inf

    if rh_pr >= min_conf:
        rh_dist = math.sqrt((bb_x - rh_x) ** 2 + (bb_y - rh_y) ** 2)
    else:
        rh_dist = math.inf

    return lh_dist, rh_dist


def compute_dist_ball_to_hands(
    ball_mask: npt.NDArray,
    person_skeleton: npt.NDArray,
    min_conf: float = 0.7,
) -> Tuple[float, float]:
    """Compute distance of ball to left and right wrists."""
    lh_y, lh_x, lh_pr = person_skeleton[get_coco_keypoint_index("left_wrist")]
    rh_y, rh_x, rh_pr = person_skeleton[get_coco_keypoint_index("right_wrist")]

    if lh_pr >= min_conf:
        lh_i, lh_j = find_closest_point_in_mask(ball_mask, lh_x, lh_y)
        lh_dist = math.sqrt((lh_i - lh_x) ** 2 + (lh_j - lh_y) ** 2)
    else:
        # If not confident in keypoint, then ignore
        lh_dist = math.inf

    if rh_pr >= min_conf:
        rh_i, rh_j = find_closest_point_in_mask(ball_mask, rh_x, rh_y)
        rh_dist = math.sqrt((rh_i - rh_x) ** 2 + (rh_j - rh_y) ** 2)
    else:
        rh_dist = math.inf

    return lh_dist, rh_dist


def is_ball_near_hand_fast(
    ball_box: BoxType,
    person_skeleton: npt.NDArray,
    dist_pixel_thres: float = 10,
    min_conf: float = 0.7,
) -> bool:
    """Similar to `is_ball_near_hand`. Since array checks are expensive
    we reduce the number of operations with a box check.
    """
    lh_y, lh_x, lh_pr = person_skeleton[get_coco_keypoint_index("left_wrist")]
    rh_y, rh_x, rh_pr = person_skeleton[get_coco_keypoint_index("right_wrist")]
    bb_x, bb_y = get_box_center(*ball_box)
    # Perform checks
    check_lh1 = lh_pr >= min_conf
    check_lh2 = math.sqrt((bb_x - lh_x) ** 2 + (bb_y - lh_y) ** 2) <= dist_pixel_thres
    check_rh1 = rh_pr >= min_conf
    check_rh2 = math.sqrt((bb_x - rh_x) ** 2 + (bb_y - rh_y) ** 2) <= dist_pixel_thres
    # Return the result
    is_near = (check_lh1 and check_lh2) or (check_rh1 and check_rh2)
    return is_near


def is_ball_near_hand(
    ball_mask: npt.NDArray,
    person_skeleton: npt.NDArray,
    dist_pixel_thres: float = 10,
    min_conf: float = 0.7,
) -> bool:
    r"""Check if a ball is sufficiently near your hand.
    :param ball_mask: Binary mask around the ball
    :param person_skeleton: Person skeleton keypoints
    :param dist_pixel_thres: Distance threshold in pixels from wrist to ball
    :param min_conf: Minimum confidence for keypoint
    """
    lh_y, lh_x, lh_pr = person_skeleton[get_coco_keypoint_index("left_wrist")]
    rh_y, rh_x, rh_pr = person_skeleton[get_coco_keypoint_index("right_wrist")]
    # Find closest point in mask
    lh_i, lh_j = find_closest_point_in_mask(ball_mask, lh_x, lh_y)
    rh_i, rh_j = find_closest_point_in_mask(ball_mask, rh_x, rh_y)
    # Perform checks
    check_lh1 = lh_pr >= min_conf
    check_lh2 = math.sqrt((lh_i - lh_x) ** 2 + (lh_j - lh_y) ** 2) <= dist_pixel_thres
    check_rh1 = rh_pr >= min_conf
    check_rh2 = math.sqrt((rh_i - rh_x) ** 2 + (rh_j - rh_y) ** 2) <= dist_pixel_thres
    # Return the result
    is_near = (check_lh1 and check_lh2) or (check_rh1 and check_rh2)
    return is_near


def find_closest_point_in_mask(
    mask: npt.NDArray, point_x: float, point_y: float
) -> List[int]:
    r"""Find closest point in a mask to a point.
    :param mask:
    :param point: [x, y]
    :return: [x, y] indices in the mask
    """
    # Find the indices of non-zero elements
    coords = np.argwhere(mask > 0)
    # Calculate the Euclidean distance between the point and all foreground coordinates
    distances = np.sqrt((coords[:, 0] - point_x) ** 2 + (coords[:, 1] - point_y) ** 2)
    # Find the index of the minimum distance
    min_index = np.argmin(distances)
    # Return the closest coordinate
    return [int(x) for x in coords[min_index]]


def merge_video_segments(*segments: SAM2SegmentsType) -> SAM2SegmentsType:
    r"""Combine arbitrary number of dictionary of dictionaries.
    :note: assumes no IDs clash
    :note: does not assume the same keys across segments
    """
    merged_segments = {}
    for segment in segments:
        for outer_key, inner_dict in segment.items():
            if outer_key not in merged_segments:
                merged_segments[outer_key] = {}
            # Combine the inner dictionaries
            merged_segments[outer_key] = {**merged_segments[outer_key], **inner_dict}
    return merged_segments


def video_segments_to_video_boxes(video_segments: SAM2SegmentsType) -> SAM2BoxesType:
    """Convert segments to bounding boxes"""
    video_boxes: SAM2BoxesType = {}
    pbar = tqdm(total=len(video_segments))
    for frame_idx, segments in video_segments.items():
        boxes: Dict[int, Optional[BoxType]] = {}
        for obj_id, obj_mask in segments.items():
            obj_box = binary_mask_to_bounding_box(obj_mask[0])
            boxes[obj_id] = obj_box
        video_boxes[frame_idx] = boxes
        pbar.update()
    pbar.close()
    return video_boxes


def label_collisions(
    video_boxes: SAM2BoxesType,
    ball_id: int,
    hoop_id: int,
    person_ids: List[int],
    interpolate_gap: int = 5,
    min_interval: int = 5,
) -> List[Tuple[int, int]]:
    """Check that persons don't conflict with the hoop & ball, and do not
    conflict with each person.
    :return: List[Tuple]
        :collision_id: ID to remove
        :collision_idx: Frame to remove items at
    """
    # Heuristic 1 - remove person IDs that consistently intersect with the ball or hoop
    person_ids_to_frames = defaultdict(lambda: [])
    for frame_idx, bboxes in video_boxes.items():
        ball_bbox = bboxes[ball_id]
        hoop_bbox = bboxes[hoop_id]
        # Remove any persons that intersect ball / hoop
        for person_id in person_ids:
            person_bbox = bboxes.get(person_id, None)
            if person_bbox is None:
                continue
            if ball_bbox is not None:
                if is_almost_same_box(person_bbox, ball_bbox):
                    person_ids_to_frames[(ball_id, person_id)].append(frame_idx)
            if hoop_bbox is not None:
                if is_almost_same_box(person_bbox, hoop_bbox):
                    person_ids_to_frames[(hoop_id, person_id)].append(frame_idx)

    # Heuristic 2 - remove person IDs that consistently intersect with others
    for frame_idx, bboxes in video_boxes.items():
        for i in range(0, len(person_ids) - 1):
            for j in range(i + 1, len(person_ids)):
                person_bbox_i = bboxes.get(person_ids[i], None)
                person_bbox_j = bboxes.get(person_ids[j], None)
                if (person_bbox_i is None) or (person_bbox_j is None):
                    continue
                if is_almost_same_box(person_bbox_i, person_bbox_j):
                    person_ids_to_frames[(person_ids[i], person_ids[j])].append(
                        frame_idx
                    )

    # Build intervals from this
    person_ids_to_frames = {
        k: find_consecutive_ranges(
            interpolate_gaps_in_numbers(v, max_gap=interpolate_gap)
        )
        for k, v in person_ids_to_frames.items()
    }
    # Remove short intervals
    person_ids_to_frames = {
        k: [l for l in v if (l[1] - l[0]) > min_interval]
        for k, v in person_ids_to_frames.items()
    }
    # Remove empty values
    person_ids_to_frames = {k: v for k, v in person_ids_to_frames.items() if len(v) > 0}

    collisions = []
    # For each of these intervals, we need to decide who to keep!
    for (id1, id2), intervals in person_ids_to_frames.items():
        start_idx = max(0, intervals[0][0] - 1)  # take right before disappearing
        box1 = video_boxes[start_idx][id1]
        box2 = video_boxes[start_idx][id2]
        size1 = compute_bbox_area(box1) if box1 is not None else 0
        size2 = compute_bbox_area(box2) if box2 is not None else 0
        # Each collision is a tuple with (1) id to overwrite, and (2) the start frame to overwrite
        if id1 == ball_id or id1 == hoop_id:
            collision = (id2, intervals[0][0])
        elif id2 == ball_id or id2 == hoop_id:
            collision = (id1, intervals[0][0])
        elif size1 >= size2:
            # Remove the smaller one
            collision = (id2, intervals[0][0])
        else:
            collision = (id1, intervals[0][0])
        # We don't need to store end b/c we just remove everything after the start
        collisions.append(collision)

    return collisions


def remove_collisions(
    video_segments: SAM2SegmentsType,
    video_boxes: SAM2BoxesType,
    collisions: List[Tuple[int, int]],
    video_width: int,
    video_height: int,
) -> Tuple[SAM2SegmentsType, SAM2BoxesType]:
    """Remove object collisions after identifying them.
    :note: edits `video_segments` and `video_boxes` in-place.
    """
    empty = np.zeros((1, video_height, video_width)).astype(bool)
    num_frames = len(video_segments)
    # Replace all entries with empty
    for person_id, start_idx in collisions:
        for frame_idx in range(start_idx, num_frames):
            video_segments[frame_idx][person_id] = empty
            video_boxes[frame_idx][person_id] = None
    return video_segments, video_boxes


def find_consecutive_ranges(numbers: List[int]) -> List[Tuple[int, int]]:
    """Convert a list of numbers into a list of consecutive ranges."""
    if len(numbers) == 0:
        return []
    numbers = sorted(numbers)
    ranges = []
    start = numbers[0]
    end = numbers[0]
    for i in range(1, len(numbers)):
        if numbers[i] == end + 1:
            end = numbers[i]
        else:
            ranges.append((start, end))
            start = end = numbers[i]
    ranges.append((start, end))
    return ranges


def interpolate_gaps_in_numbers(numbers: List[int], max_gap: int = 3) -> List[int]:
    """Interpolates gaps in a list of numbers where the gap size is less than `max_gap`."""
    if len(numbers) == 0:
        return []
    numbers = sorted(numbers)
    interpolated = [numbers[0]]

    for i in range(1, len(numbers)):
        current = numbers[i]
        previous = interpolated[-1]
        gap = current - previous

        if 1 < gap <= max_gap:
            interpolated.extend(range(previous + 1, current))

        interpolated.append(current)

    return interpolated


def is_almost_same_box(box1: BoxType, box2: BoxType, eps: int = 10) -> bool:
    """Is almost the same bounding box."""
    return (
        (abs(box1[0] - box2[0]) <= eps)
        and (abs(box1[1] - box2[1]) <= eps)
        and (abs(box1[2] - box2[2]) <= eps)
        and (abs(box1[3] - box2[3]) <= eps)
    )


def to_json(obj: Dict | List, filepath: str):
    with open(filepath, "w") as fp:
        json.dump(obj, fp)


def from_json(filepath: str) -> Dict | List:
    with open(filepath, "r") as fp:
        data = json.load(fp)
    return data


def to_jsonlines(obj: List, filepath: str):
    with jsonlines.open(filepath, "w") as writer:
        writer.write_all(obj)


def from_jsonlines(filepath: str) -> List[Dict]:
    with jsonlines.open(filepath, "r") as reader:
        obj = []
        for row in reader:
            obj.append(row)
    return obj


def from_yaml(filepath: str):
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    return config


def tick() -> int:
    r"""Helper function to get current time."""
    return int(time.time())


def create_logger(log_name: str, log_file: str) -> logging.Logger:
    r"""Create a logger that we can use for both printing to standard output
    and to a log file.
    :param log_name:
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Stream handler for printing to console (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def compress_video_segments(video_segments: SAM2SegmentsType) -> CompressedSegmentsType:
    r"""Compress video segments for uploading into the cloud.
    :note: this converts masks from arrays to paths (lossy)
    """
    compressed_segments = {}
    for frame_idx, segments in video_segments.items():
        compressed = {}
        for obj_id, segment_mask in segments.items():
            compressed[obj_id] = binary_mask_to_coco_format(segment_mask[0])
        compressed_segments[frame_idx] = compressed
    return compressed_segments


def decompress_video_segments(
    compressed_segments: CompressedSegmentsType,
    video_width: int,
    video_height: int,
) -> SAM2SegmentsType:
    r"""Decompress video segments after downloading from the cloud.
    :note: this converts saved paths to binary masks (increase in memory)
    """
    video_segments = {}
    for frame_idx, segments in compressed_segments.items():
        decompressed = {}
        for obj_id, segment_path in segments.items():
            segment_mask = coco_format_to_binary_mask(
                segment_path, video_width, video_height
            )
            segment_mask = segment_mask[np.newaxis, ...]  # add a dim
            segment_mask = segment_mask.astype(bool)
            decompressed[obj_id] = segment_mask
        video_segments[frame_idx] = decompressed
    return video_segments


def fig2img(fig) -> npt.NDArray:
    r"""Make an image from a figure."""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    # Ensure the buffer size matches the expected size (width * height * 3 for RGB)
    if buf.size != width * height * 4:
        raise ValueError(
            f"Buffer size {buf.size} does not match the expected size {width * height * 4}."
        )
    # Reshape the buffer to match the dimensions of the figure
    buf = buf.reshape(height, width, 4)
    buf = buf[:, :, 1:]
    return buf


def plot_bbox(
    box: BoxType,
    ax: Axes,
    color: str,
    label: str,
    linewidth: int = 2,
    linestyle: str = "solid",
):
    """Plot a bounding box on the figure."""
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    # Add rectangle to plot
    rect = patches.Rectangle(
        (x1, y1),
        w,
        h,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    # Add label text
    ax.text(x1, y1, label, fontsize=8, color=color, verticalalignment="top")


def format_bbox(x: BoxType) -> BoxType:
    r"""Standardize the format for a bounding box to make all points integers,
    and make the probability a floating point number.
    """
    bbox = [
        int(round(x[0])),
        int(round(x[1])),
        int(round(x[2])),
        int(round(x[3])),
    ]
    if len(x) == 5:
        bbox.append(float(round(x[4], 3)))  # type: ignore
    return bbox


def get_resolution_dimensions(name: str) -> Tuple[int, int]:
    r"""Convert a video resolution code to a tuple of width and height.
    :param name (str): Video resolution code
    :return (Tuple[int, int]): Tuple of width and height
    """
    if name == "144p":
        return (256, 144)
    elif name == "240p":
        return (426, 240)
    elif name == "360p":
        return (640, 360)
    elif name == "480p":
        return (640, 480)
    elif name == "720p":
        return (1280, 720)
    elif name == "1080p":
        return (1920, 1080)
    else:
        raise ValueError(f"Invalid resolution code: {name}")


def transcode_to_common_fps(input_file: str, output_file: str, fps: float = 30.0):
    (
        ffmpeg.input(input_file)
        .output(
            output_file,
            r=fps,
            vcodec="libx264",
            crf=24,
            preset="slow",
            acodec="copy",
            movflags="faststart",
        )
        .run(overwrite_output=True)
    )


def transcode_to_optimal_size(
    input_file: str,
    output_file: str,
    data_dir: str,
    preset: str = "superfast",
    resolution: str = "720p",
    quiet: bool = False,
):
    r"""This function compresses a video file. It does not do resolution change atm."""
    width, height = get_resolution_dimensions(resolution)
    scale = f"scale={width}:{height}"
    (
        ffmpeg.input(input_file)
        .output(
            join(data_dir, output_file),
            vcodec="libx264",
            crf=28,
            movflags="faststart",
            preset=preset,
            r=30,
            vf=scale,
        )
        .run(overwrite_output=True, quiet=quiet)
    )


def transcode_to_multiple_sizes(
    input_file: str,
    output_file_base: str,
    data_dir: str,
    preset: str = "superfast",
    resolutions: List[str] = ["1080p", "720p", "360p", "144p"],
    quiet: bool = False,
):
    """Compress a video file to multiple resolutions.
    :note: we do this with a single ffmpeg command to avoid re-encoding
    :note: this standardizes video and audio encodings
    :param input_file (str): Input video file
    :param data_dir (str): Data directory
    :param preset (str): FFMPEG preset
    :param resolutions (List[str]): List of resolutions to transcode to
    """
    outputs = []

    # Define input stream
    video_stream = ffmpeg.input(input_file)
    audio_stream = video_stream.audio

    # Define filter and output for each resolution
    for resolution in resolutions:
        width, height = get_resolution_dimensions(resolution)
        # Apply scale filter
        scaled_stream = video_stream.filter("scale", width, height)
        output_path = join(data_dir, f"{output_file_base}_{resolution}.mp4")
        outputs.append(
            ffmpeg.output(
                scaled_stream,
                audio_stream,
                output_path,
                vcodec="libx264",
                crf=28,
                movflags="faststart",
                preset=preset,
                r=30,
                acodec="aac",
            )
        )
    # Run the ffmpeg command
    ffmpeg.merge_outputs(*outputs).run(overwrite_output=True, quiet=quiet)



def check_if_video_has_audio_ffmpeg(video_file: str) -> bool:
    r"""Check if video has audio.
    :param video_file (str): Video file
    """
    p = ffmpeg.probe(video_file, select_streams="a")
    # If p['streams'] is not empty, clip has an audio stream
    if p["streams"] is None:
        return False

    if len(p["streams"]) == 0:
        return False

    return True


def extract_subclip_ffmpeg(
    video_file: str,
    start_ts: float,
    end_ts: float,
    out_file: str,
    quiet: bool = True,
):
    """
    :param video_file (str): Video file to extract subclip from
    :param start_ts (float): Start timestamp
    :param end_ts (float): End timestamp
    :param out_file (str): Output video file
    """
    assert start_ts < end_ts, "Start timestamp must be less than end timestamp"
    try:
        (
            ffmpeg.input(video_file, ss=start_ts, t=end_ts - start_ts)
            .output(out_file, codec="copy")
            .run(overwrite_output=True, quiet=quiet)
        )
    except ffmpeg.Error as e:
        raise e


def extract_subclip_ffmpeg_reencode_fps(
    video_file: str,
    start_ts: float,
    end_ts: float,
    out_file: str,
    out_fps: int = 30,
    quiet: bool = True,
):
    r"""Extract a subclip from a video file + reencode FPS"""
    assert start_ts < end_ts, "Start timestamp must be less than end timestamp"
    try:
        (
            ffmpeg.input(video_file, ss=start_ts, t=end_ts - start_ts)
            .output(
                out_file,
                vcodec="libx264",
                crf=23,
                preset="fast",
                r=out_fps,
                acodec="copy",
            )
            .run(overwrite_output=True, quiet=quiet)
        )
    except ffmpeg.Error as e:
        raise e


def extract_audio_ffmpeg(video_file: str, out_file: str) -> str:
    r"""Extract the audio from a segment of the video file.
    :param start_time (int)
    :param end_time (int)
    """
    assert ".mp3" in out_file and ".mp4" in video_file
    ffmpeg.input(video_file).output(
        out_file, **{"qscale:a": 0, "map": "a"}
    ).overwrite_output().run()
    return out_file


def stitch_video_and_audio_ffmpeg(
    video_file: str,
    audio_file: str,
    out_file: str,
) -> str:
    r"""Stitch together the video and audio files.
    We assume both are the same length.
    """
    assert ".mp3" in audio_file and ".mp4" in video_file and ".mp4" in out_file
    video_stream = ffmpeg.input(video_file)
    audio_stream = ffmpeg.input(audio_file)
    ffmpeg.output(
        video_stream,
        audio_stream,
        out_file,
        vcodec="copy",
        acodec="aac",
        strict="experimental",
    ).overwrite_output().run()
    return out_file


def stitch_videos_ffmpeg(
    video_files: List[str],
    out_path: str,
    cache_dir: str = "./cache",
    fps: int = 30,
) -> str:
    r"""Use FFMPEG to stitch video clips together."""
    transcoded_files: List[str] = []
    for i, video_file in enumerate(video_files):
        transcoded_file = join(cache_dir, f"transcoded_{i}.mp4")
        transcode_to_common_fps(video_file, transcoded_file, fps=fps)
        transcoded_files.append(transcoded_file)

    tmp_file = realpath(join(cache_dir, "input.txt"))

    # Create a temporary file to list videos
    with open(tmp_file, "w") as f:
        for filename in transcoded_files:
            filename = realpath(filename)
            f.write(f"file '{filename}'\n")

    # Run ffmpeg command to concatenate videos without re-encoding
    ffmpeg.input(tmp_file, format="concat", safe=0).output(
        out_path, c="copy", vsync=2, r=fps
    ).run(overwrite_output=True)

    # Delete the temporary file
    if exists(tmp_file):
        remove(tmp_file)

    # Delete the transcodings
    for transcoded_file in transcoded_files:
        if exists(transcoded_file):
            remove(transcoded_file)

    return out_path


def reencode_video_mp4v_to_h264(video_file: str, out_file: str):
    r"""Reencode a video from mp4v to h264.
    :param video_file (str): Video file to reencode
    :param out_file (str): Output video file
    """
    (
        ffmpeg.input(video_file)
        .output(out_file, vcodec="libx264", acodec="copy")
        .run(overwrite_output=True)
    )


def get_video_avg_fps(video_file: str) -> float:
    r"""Get average frames per second of a video.
    @note warning: FPS is an illusion
    :param video_file (str): Video file to get fps for
    @return fps (float): Frames per second
    """
    reader = cv2.VideoCapture(video_file)
    assert reader.isOpened(), f"Could not open video file {video_file}"
    fps = reader.get(cv2.CAP_PROP_FPS)
    reader.release()
    return fps


def get_video_frame_size(video_file: str) -> Tuple[int, int]:
    reader = cv2.VideoCapture(video_file)
    assert reader.isOpened(), f"Could not open video file {video_file}"
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reader.release()
    return width, height


def get_video_length(video_file: str) -> float:
    duration = ffmpeg.probe(video_file)["format"]["duration"]
    return float(duration) if len(duration) > 0 else 0.0


def round_fps_to_nearest_30(fps: float) -> int:
    """Rounds the given fps value to the nearest multiple of 30.
    :param fps: The frames per second value to round.
    :return: The fps value rounded to the nearest multiple of 30.
    """
    return int(round(fps / 30) * 30)


def list_to_comma_sep_str(x: List[int] | List[float]) -> str:
    r"""Join the list items into a string separated by commas
    :param x: List of variables
    :return: Comma-separated string
    """
    return ",".join(map(str, x))


def comma_sep_str_to_list(
    s: str, as_int: bool = False, as_bool: bool = False
) -> Union[List[float], List[int]]:
    r"""Parse a string to a list of floats.
    @param as_int (bool, default=False): If false, use float
    """
    outputs: List[float] = []
    if len(s) > 0:
        parts: List[str] = s.split(",")
        for part in parts:
            try:
                x = float(part)  # cast to numeric
                if as_int:
                    x = int(round(x))  # cast to int
                elif as_bool:
                    x = bool(x)
                outputs.append(x)
            except Exception:
                continue
    return outputs


def optimize_paint_keypoints_from_mask(
    paint_mask: npt.NDArray,
    paint_box: npt.NDArray,
    eps: float = 2.0,
) -> npt.NDArray:
    r"""Returns the four corners of the paint.
    :param paint_mask: Segmentation mask for paint
    :param paint_box: Bounding box for paint
    :return: Paint keypoints
    """
    a, b = paint_mask.shape
    x1, y1, x2, y2 = paint_box

    # Help converge better. Should make less hacky
    x1 += 10
    y1 += 10
    x2 -= 10
    y2 -= 10

    # Initial point
    paint = np.array([x1, y1, x1, y2, x2, y2, x2, y1])

    def loss(x):
        r"""Loss function."""
        img = Image.new("L", (b, a), 0)
        ImageDraw.Draw(img).polygon(x.tolist(), outline=1, fill=1)
        mask = np.array(img)
        return np.count_nonzero(mask ^ paint_mask)

    # Perform an optimization
    result = sp.optimize.minimize(loss, paint, options={"eps": eps})
    return result.x


def get_median_background(
    frame_dir: str,
    frame_names: List[str],
    out_file: str,
    interval: int = 10,
):
    r"""Returns image with median pixel value.
    :param frame_dir: Directory of frame images
    :param frame_names: List of frame names
    :param interval: Interval to sample frames
    """
    # Get samples and create background file
    background = np.median(
        np.stack(
            [
                load_image(join(frame_dir, frame_name))
                for frame_name in frame_names[::interval]
            ]
        ),
        axis=0,
    )
    cv2.imwrite(out_file, background)


def compute_pairwise_distances(vectors: npt.NDArray) -> npt.NDArray:
    """Computes all pairwise distances between a list of vectors."""
    # Convert input to a numpy array if it's not already
    vectors = np.asarray(vectors)
    # Compute the pairwise Euclidean distances
    distances = cdist(vectors, vectors, metric="euclidean")
    return distances


def generate_uuid() -> str:
    unique_id = uuid.uuid4()
    return str(unique_id)


def visualize_hoop_on_frame(
    image_path: str,
    hoop_box: BoxType,
    backboard_box: BoxType,
    out_path: str,
    dpi: int = 100,
):
    r"""Visualize the hoop & backboard on the image."""
    im = Image.open(image_path)
    width, height = im.size
    figsize = (width / dpi, height / dpi)

    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im) # type: ignore
    ax.axis("off") # type: ignore

    rect1 = patches.Rectangle(
        (hoop_box[0], hoop_box[1]),
        hoop_box[2] - hoop_box[0],
        hoop_box[3] - hoop_box[1],
        linewidth=2,
        edgecolor="#F05A7E",
        facecolor="none",
    )
    ax.add_patch(rect1) # type: ignore
    ax.text( # type: ignore
        hoop_box[0],
        hoop_box[1] - 10,
        "hoop",
        fontsize=12,
        color="#F05A7E",
        fontweight="bold",
    )
    rect2 = patches.Rectangle(
        (backboard_box[0], backboard_box[1]),
        backboard_box[2] - backboard_box[0],
        backboard_box[3] - backboard_box[1],
        linewidth=2,
        edgecolor="#0B8494",
        facecolor="none",
    )
    ax.add_patch(rect2) # type: ignore
    ax.text( # type: ignore
        backboard_box[0],
        backboard_box[1] - 10,
        "backboard",
        fontsize=12,
        color="#0B8494",
        fontweight="bold",
    )

    plt.savefig(out_path)


def visualize_ball_on_frame(
    image_path: str,
    out_path: str,
    points: Optional[npt.NDArray] = None,
    labels: Optional[npt.NDArray] = None,
    box: Optional[npt.NDArray] = None,
    plot_title: Optional[str] = None,
):
    r"""Visualize the ball on an image frame.
    :param points: Points to visualize
    :param labels: Labels to visualize
    :param box: Box to visualize
    """
    im = Image.open(image_path)
    width, height = im.size
    figsize = (width / 100, height / 100)
    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im) # type: ignore
    ax.axis("off") # type: ignore
    if plot_title:
        ax.set_title(plot_title) # type: ignore
    if points is not None and labels is not None:
        show_points(points, labels, ax) # type: ignore
    elif box is not None:
        box_: BoxType = [box[0], box[1], box[2], box[3]]
        show_bbox(box_, ax, "#ffc300", label="ball") # type: ignore
    plt.savefig(out_path)


def visualize_persons_on_frame(
    image_path: str,
    person_prompts: List[PromptType],
    out_path: str,
    plot_title: Optional[str] = None,
):
    r"""Visualize person masks on the image frame."""
    im = Image.open(image_path)
    width, height = im.size
    figsize = (width / 100, height / 100)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im) # type: ignore
    ax.axis("off") # type: ignore
    if plot_title:
        ax.set_title(plot_title) # type: ignore
    for _, _, points, labels, _ in person_prompts:
        if points is not None and labels is not None:
            show_points(points, labels, ax) # type: ignore
    plt.savefig(out_path)


def is_url_or_file(path_or_url: str) -> int:
    """Check if an input is a url or a file
    :return 0 if url, 1 if file, -1 if neither
    """
    # Define a regex for matching a URL
    url_pattern = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https:// or ftp:// or ftps://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|"  # ...or IPv4
        r"\[?[A-F0-9]*:[A-F0-9:]+\]?)"  # ...or IPv6
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    # Check if the input matches a URL pattern
    if re.match(url_pattern, path_or_url):
        return 0
    # Check if the input is a valid local file path
    if isfile(path_or_url):
        return 1
    # If neither, return 'Invalid'
    return -1


def get_keyframe_timestamps(
    paused_ts: Union[List[float], List[int]],
    video_length: int | float,
    min_diff_ts: int = 5,
    nearby_ts: int = 1,
) -> List[Dict[str, float]]:
    """Get keyframe timestamps from paused timestamps.
    :note These are the frames for annotating paint and also admin panel.
    :note we make a few samples per index
    :param paused_ts (List[float]): Paused timestamps
    :param video_length (int): Length of the video
    :return (List[Dict]): List of dictionaries with keys {'ts', 'idx', 'start_ts', 'end_ts'}
    """
    # Filter the timestamps
    filtered_ts: List[float] = [0]
    for i in range(len(paused_ts)):
        if paused_ts[i] - filtered_ts[-1] >= min_diff_ts:
            filtered_ts.append(paused_ts[i])
    filtered_ts.append(video_length)
    # Generate samples
    samples = []
    last = 0
    for i in range(1, len(filtered_ts)):
        # Calculate the mid-point between two timestamps
        sample = round((last + filtered_ts[i]) / 2.0, 1)
        # Add timestamps around the paused timestamp
        if sample - nearby_ts > 0:
            samples.append(
                {
                    "ts": sample - nearby_ts,
                    "idx": i - 1,
                    "start_ts": last,
                    "end_ts": filtered_ts[i],
                }
            )
        samples.append(
            {"ts": sample, "idx": i - 1, "start_ts": last, "end_ts": filtered_ts[i]}
        )
        if sample + nearby_ts < video_length:
            samples.append(
                {
                    "ts": sample + nearby_ts,
                    "idx": i - 1,
                    "start_ts": last,
                    "end_ts": filtered_ts[i],
                }
            )
        # Move to the next timestamp
        last = filtered_ts[i]
    return samples


def get_image_shape(image_path: str) -> Tuple[int, int]:
    """Get width and height of image."""
    img = Image.open(image_path)
    return img.size


def truncate_and_crop_clip(
    clip_path: str,
    box: BoxType,
    start_ts: float,
    end_ts: float,
    out_file: str,
    out_fps: int = 30,
):
    """Truncate and crop the clip to be near the hoop.
    @param clip_path: The path to the clip
    @param box: The hoop location
    @param start_ts: Start timestamp to truncate
    @param end_ts: End timestamp to truncate
    @param out_file: The file to save the output
    @param out_fps: output FPS (default: 30)
    """
    assert start_ts < end_ts, "Start timestamp must be less than end timestamp"
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    (
        ffmpeg.input(clip_path, ss=start_ts, to=end_ts)
        .filter("crop", w, h, x1, y1)
        .output(
            out_file, vcodec="libx264", crf=23, preset="fast", r=out_fps, acodec="copy"
        )
        .run(overwrite_output=True, quiet=True)
    )


def truncate_clip_ffmpeg(
    clip_path: str, start_ts: float, end_ts: float, out_file: str, out_fps: int = 30
):
    """Truncate clip and normalize FPS."""
    (
        ffmpeg.input(clip_path, ss=start_ts, to=end_ts)
        .output(
            out_file, vcodec="libx264", crf=23, preset="fast", r=out_fps, acodec="copy"
        )
        .run(overwrite_output=True, quiet=True)
    )
