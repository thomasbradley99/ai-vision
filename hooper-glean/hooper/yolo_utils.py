import cv2
import torch
import numpy.typing as npt
from ultralytics import YOLO
from typing import Tuple, List

from shared_utils.types import BoxType, HoopOutput

from .utils import load_image, format_bbox
from .ml_utils import get_shot_box_from_hoop_box


def load_yolo_model(weight_file: str, device_name: str = "cuda:0"):
    model = YOLO(weight_file)
    model.fuse()  # about 5-10% faster
    model.to(device_name)
    return model


def calculate_optimal_batch_size(
    frame_shape: Tuple[int, int, int], available_memory_gb: float
) -> int:
    """Calculate a reasonable batch size based on frame dimensions and available GPU memory"""
    h, w, c = frame_shape

    # Memory needed per frame (more conservative estimate)
    # Account for input frame, feature maps, and model overhead
    bytes_per_frame = h * w * c * 4  # Base size (float32)
    model_overhead_factor = 10  # YOLO models have significant memory overhead

    # Total bytes per frame with overhead
    total_bytes_per_frame = bytes_per_frame * model_overhead_factor

    # Convert to GB
    gb_per_frame = total_bytes_per_frame / (1024**3)

    # Calculate max frames that can fit in 80% of available memory
    max_frames = int((available_memory_gb * 0.8) / gb_per_frame)

    # Add safety margin and cap at reasonable values
    safe_batch_size = max(1, min(max_frames, 64))  # Cap at 64 as practical upper limit

    return int(safe_batch_size)


def get_gpu_memory() -> float:
    """Returns available GPU memory in GB"""
    if not torch.cuda.is_available():
        return 0
    # Get device properties
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    # Get memory information
    total_memory = props.total_memory / (1024**3)  # Convert to GB
    # Get current memory usage
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    # Calculate available memory
    available = total_memory - allocated
    return available


def fetch_batch_of_frames(
    cap: cv2.VideoCapture,
    batch_size: int,
    vid_stride: int = 1,
    cast_to_rgb: bool = True,
) -> Tuple[List[npt.NDArray], List[float]]:
    frames = []
    timestamps = []
    for _ in range(batch_size):
        # Skip frames according to stride
        for _ in range(vid_stride - 1):
            # Much faster than read() when we don't need the frame
            cap.grab()
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB if requested
        if cast_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_ts = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)
        timestamps.append(frame_ts)
    assert len(frames) == len(timestamps)
    return frames, timestamps


def detect_balls_with_yolo(
    model: YOLO,
    image_paths: List[str],
    hoops: List[HoopOutput],
    min_conf: float = 0.5,
    crop_size: int = 256,
    device_name: str = "cuda:0",
) -> List[List[BoxType]]:
    """Batched ball detection using a finetuned YOLOv11.
    This model acts as a backup since we know that it is prone to
    false negatives and false positives.
    @param image_paths: List of paths to frames
    @param min_conf: Score threshold for detecting a bounding box
    @param crop_size: Size of the crop around the hoop
    @return boxes: [[x1, y1, x2, y2, score]...]
    """
    assert len(hoops) == len(image_paths)
    crops = []
    for hoop, path in zip(hoops, image_paths):
        image = load_image(path, as_rgb=True)
        crop_box = get_shot_box_from_hoop_box(hoop.box, crop_size)
        crop = image[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
        crops.append(crop)
    results = model(
        crops,
        conf=min_conf,
        imgsz=crop_size,
        verbose=False,
        device=device_name,
    )
    preds = []
    for i in range(len(results)):
        crop_box_i = get_shot_box_from_hoop_box(hoops[i].box, crop_size)
        boxes = results[i].boxes
        preds_i = []
        for j in range(len(boxes)):
            prob_j = round(float(boxes[j].conf.item()), 3)
            raw_box_j = boxes[j].xyxy[0].tolist()
            # Compute global coordinates from the box
            box_j = [
                int(raw_box_j[0] + crop_box_i[0]),
                int(raw_box_j[1] + crop_box_i[1]),
                int(raw_box_j[2] + crop_box_i[0]),
                int(raw_box_j[3] + crop_box_i[1]),
            ]
            pred = box_j + [prob_j]
            preds_i.append(format_bbox(pred))
        preds.append(preds_i)
    return preds
