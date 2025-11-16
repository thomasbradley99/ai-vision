import torch
from PIL import Image
from typing import List, Tuple
from transformers.image_processing_utils import BaseImageProcessor
from transformers import DFineForObjectDetection, AutoImageProcessor
from shared_utils.types import BoxType
from .utils import format_bbox


def load_dfine_model(
    model_name: str = "ustc-community/dfine-xlarge-obj365",
    device_name: str = "cuda:0",
) -> Tuple[torch.nn.Module, BaseImageProcessor]:
    device = torch.device(device_name)
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = DFineForObjectDetection.from_pretrained(model_name)
    model = model.to(device)  # type: ignore
    model = model.eval()
    return model, processor


def detect_balls_with_dfine(
    model: torch.nn.Module,
    processor: BaseImageProcessor,
    image_paths: List[str],
    label_ids: List[int] = [178],  # 177+1 for 1-indexed
    min_conf: float = 0.5,
    device_name: str = "cuda",
) -> List[List[BoxType]]:
    """Batched ball detection using DFine pretrained on Object 365.
    @param image_paths: List of paths to frames
    @param box_threshold: Score threshold for detecting a bounding box
    @return boxes: [[x1, y1, x2, y2, score]...]
    """
    device = torch.device(device_name)
    pil_images = [Image.open(path) for path in image_paths]
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # compute sizes - important to reverse
    target_sizes = torch.tensor([img.size[::-1] for img in pil_images], device=device)
    results = processor.post_process_object_detection(  # type: ignore
        outputs,
        target_sizes=target_sizes,
        threshold=min_conf,
    )
    preds = []
    for i in range(len(results)):
        result = results[i]
        preds_i = []
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            score, label = score.item(), label_id.item()
            if label not in label_ids:
                continue
            pred = box.tolist() + [round(score, 3)]
            preds_i.append(format_bbox(pred))
        preds.append(preds_i)
    return preds


def get_hoop_width(hoop_box: BoxType) -> float:
    """Do this in a way that might support vertical videos."""
    width = hoop_box[2] - hoop_box[0]
    height = hoop_box[3] - hoop_box[1]
    return max(width, height)


def get_ball_diameter(ball_box: BoxType) -> float:
    """Diameter will be the larger between height and width
    to handle some distortion
    """
    width = ball_box[2] - ball_box[0]
    height = ball_box[3] - ball_box[1]
    return max(width, height)
